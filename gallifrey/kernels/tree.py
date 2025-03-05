from __future__ import annotations

from functools import partial

import beartype.typing as tp
import jax.numpy as jnp
from equinox.internal import while_loop as bounded_while_loop
from flax import nnx
from jax import jit, lax
from jaxtyping import Bool, Float, Int

from gallifrey.kernels.atoms import AbstractAtom, AbstractOperator
from gallifrey.kernels.library import KernelLibrary
from gallifrey.parameter import KernelParameter
from gallifrey.utils.tree_helper import (
    calculate_max_leaves,
    calculate_max_nodes,
    calculate_max_stack_size,
    get_child_idx,
    get_depth,
    get_parameter_leaf_idx,
)
from gallifrey.utils.typing import ScalarBool, ScalarFloat, ScalarInt


class TreeKernel(nnx.Module):
    """
    A kernel class to evaluate a tree-like kernel expression. The tree expression
    is a jnp array, with the tree structure encoded as a level-order array.
    Similar to NestedCombinationKernel in GPJax, but with explicit tree
    structure.

    Assume for example that the tree expression is [2, 1, 2, -1, -1, 0, 1]
    and the kernel library is [RBFAtom, LinearAtom, SumAtom, ProductAtom]. In this case
    -1 -> empty node,
    0 -> RBFKernel,
    1 -> LinearKernel,
    2 -> Sum,
    3 -> Product.
    The is_operator array should be [False, False, True, True].
    The tree expression would then be evaluated to
    Sum(Linear(x, y), Sum(RBF(x,y), Linear(x,y)).

    The same kernel could be constructed using NestedCombinationKernel, but the
    tree structure would not be explicit.

    Attributes
    (NOTE: A lot of values are initialized as nnx.Variable, use .value to access the
    actual value.)
    ----------
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, a level-order array describing the tree structure.
        Negative values indicate empty nodes.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree expression.
    max_depth : int
        The maximum depth of the tree expression.
    max_stack : ScalarInt
        The maximum size of the stack used to evaluate the tree expression,
        which is equal to the maximum depth of the tree expression + 1.
    max_leaves : ScalarInt
        The maximum number of leaves in the tree expression.
    max_atom_parameters : ScalarInt
        The maximum number of parameters of any atom in the atom library. Comes
        from the kernel library.
    max_total_parameters : ScalarInt
        The maximum number of parameters in the tree expression (max_leaves
        * max_atom_parameters).
    num_datapoints : ScalarInt
        The number of datapoints in the input training data. (This is needed to
        correctly construct the gram matrix, in a jit-compatible way.)
    root_idx : ScalarInt
        The index of the root node in the tree expression.
    atoms : tuple[AbstractAtom]
        The atoms in the kernel library.
    operators : tuple[AbstractOperator]
        The operators in the kernel library.
    num_atoms : ScalarInt
        The number of atoms in the kernel library.
    num_operators : ScalarInt
        The number of operators in the kernel library.
    is_operator : Bool[jnp.ndarray, " D"]
        A boolean array indicating whether the node in the tree expression is an
        operator or not. Comes from the kernel library.
    post_order_expression : Int[jnp.ndarray, " D"]
        The tree_expression in post-order traversal notation. Used for efficient
        evaluation of the tree expression. Negative values indicate empty nodes.
    post_level_map : Int[jnp.ndarray, " D"]
        A map from the post-order index to the level-order index. The value at
        a given index in the post-order expression corresponds to the index of
        the same node in the level-order expression. Negative values indicate
        empty nodes.
    num_nodes : ScalarInt
        The actual number of nodes in the tree expression.
    node_sizes : Int[jnp.ndarray, " D"]
        An array containing the sizes of the nodes in the tree. The size of a node
        is the number of nodes in the subtree rooted at that node. The size of a
        given node is the value at its index in the level-order expression.
    node_heights : Int[jnp.ndarray, " D"]
        An array containing the heights of the nodes in the tree. The height of a node
        is the length of the longest path from the node to a leaf node (i.e. the heigh
        of a leaf node is 0). The height of a given node is the value at its index in
        the level-order expression.
    parameters : KernelParameter(Float[jnp.ndarray, "M N"])
        KernelParameter instance, that holds a 2D array of kernel parameters with shape
        (max_leaves, max_atom_parameters). The parameters are used to evaluate the tree
        kernel.
    leaf_level_map : Int[jnp.ndarray, " N"]
        A map from the leaf index to the level-order index. The leaf index i corresponds
        to the i-th entry in the parameters 0th axis. The value of the leaf_level_map
        at index i is the index of the node with these parameters in the level-order
        expression.
        NOTE: The leaf index is decided based on the depth of the tree, and might not be
        in a simple order. See 'gallifrey.utils.tree_helper.get_parameter_leaf_idx' for
        more information.

    """

    def __init__(
        self,
        tree_expression: Int[jnp.ndarray, " D"],
        kernel_library: KernelLibrary,
        max_depth: int,
        num_datapoints: int,
        root_idx: Int[jnp.ndarray, ""] = jnp.array(0),
    ):
        """
        Initialize a tree kernel.

        Parameters
        ----------
        tree_expression : Int[jnp.ndarray, " D"]
            The tree expression, a level-order expression of the tree.
        kernel_library : KernelLibrary
            An instance of the KernelLibrary class, containing the atoms
            and operators to evaluate the tree expression.
        max_depth : int
            The maximum depth of the tree expression. The length of the
            tree expression should be 2^(max_depth+1) - 1.
        num_datapoints : int, optional
            The number of datapoints in the input data. (This is needed to
            correctly construct the gram matrix.)
        root_idx : Int[jnp.ndarray, ""], optional
            The index of the root node in the tree expression. By default 0.

        """
        self.init_tree(
            tree_expression,
            kernel_library,
            max_depth,
            num_datapoints,
            root_idx,
        )
        self.init_parameters()

    def init_tree(
        self,
        tree_expression: Int[jnp.ndarray, " D"],
        kernel_library: KernelLibrary,
        max_depth: int,
        num_datapoints: int,
        root_idx: Int[jnp.ndarray, ""] = jnp.array(0),
    ) -> None:
        """
        Initialize the base attributes of the tree kernel, which means
        setting the
        - tree_expression,
        - max_nodes,
        - root_idx,
        - max_depth,
        - is_operator (from the kernel library),
        - atom_library (from the kernel library),
        - max_atom_parameters (from the kernel library),
        - post_order_expression,
        - post_level_map,
        - num_nodes,
        - node_sizes, and
        - node_heights.

        The pre-order traversal expression and map attributes are initialized
        but set to None. They get filled when calling the 'display' method.

        Parameters
        ----------
        tree_expression : Int[jnp.ndarray, " D"]
            The tree expression, a level-order expression of the tree.
        kernel_library : tp.Optional[KernelLibrary], optional
            An instance of the KernelLibrary class, containing the atoms and
            operators to evaluate the tree expression.
        max_depth : int
            The maximum depth of the tree expression. The length of the
            tree expression should be 2^(max_depth+1) - 1.
        num_datapoints : int, optional
            The number of datapoints in the input data. (This is needed to
            correctly construct the gram matrix.)
        root_idx : Int[jnp.ndarray, ""], optional
            The index of the root node in the tree expression. By default 0.

        """
        # set base attributes
        self.tree_expression = nnx.Variable(tree_expression)

        self.max_depth = nnx.Variable(max_depth)
        self.max_stack = nnx.Variable(calculate_max_stack_size(max_depth))
        self.max_nodes = nnx.Variable(calculate_max_nodes(max_depth))
        self.max_leaves = nnx.Variable(calculate_max_leaves(max_depth))

        self.num_datapoints = nnx.Variable(num_datapoints)

        self.root_idx = nnx.Variable(root_idx)

        # check if the root index is valid
        # if self.root_idx.value > self.max_nodes - 1:
        #     raise ValueError(
        #         "Root index must be less than the maximum number of nodes."
        #     )

        # get the kernel library attributes
        self.atoms = tuple(kernel_library.atoms)
        self.operators = tuple(kernel_library.operators)
        self.num_atoms = nnx.Variable(kernel_library.num_atoms)
        self.num_operators = nnx.Variable(kernel_library.num_operators)
        self.is_operator = nnx.Variable(kernel_library.is_operator)
        self.max_atom_parameters = nnx.Variable(kernel_library.max_atom_parameters)
        self.max_total_parameters = nnx.Variable(
            self.max_leaves.value * self.max_atom_parameters.value  # type: ignore
        )

        # run the post-order traversal and get node metrics
        (
            post_order_expression,
            num_nodes,
            post_level_map,
            node_sizes,
            node_heights,
            leaf_level_map,
        ) = self.get_post_order_and_node_metrics()

        self.post_order_expression = nnx.Variable(post_order_expression)
        self.post_level_map = nnx.Variable(post_level_map)
        self.num_nodes = nnx.Variable(num_nodes)
        self.node_sizes = nnx.Variable(node_sizes)
        self.node_heights = nnx.Variable(node_heights)
        self.leaf_level_map = nnx.Variable(leaf_level_map)

    def init_parameters(self) -> None:
        """
        Initialize the parameters of the tree kernel. The parameters are
        a KernelParameter instance, which holds a 2D array of kernel parameters
        with shape (max_leaves, max_atom_parameters).
        """
        # initialise parameters
        self.parameters = KernelParameter(
            jnp.ones(
                (
                    self.max_leaves.value,
                    self.max_atom_parameters.value,
                )
            )
        )

    def __call__(
        self,
        x: Float[jnp.ndarray, " D"],
        y: Float[jnp.ndarray, " D"],
    ) -> ScalarFloat:
        """
        Evaluate the tree kernel by traversing the tree expression.

        Parameters
        ----------
        x : Float[jnp.ndarray, " D"]
            Input x data.
        y : Float[jnp.ndarray]
            Input y data.

        Returns
        -------
        ScalarFloat
            The kernel value.
        """
        return _evaluate_tree(
            jnp.atleast_1d(x),
            jnp.atleast_1d(y),
            self.post_order_expression.value,
            self.post_level_map.value,
            self.is_operator.value,
            self.parameters.value,
            self.atoms,
            self.operators,
            (1, 1),
            int(self.num_atoms.value),
            int(self.max_depth.value),
        ).squeeze()

    def _tree_viz(self, include_parameters: bool = True) -> str:
        """
        Helper method for __str__ with optional parameter to include the
        kernel parameters in the visualization.

        Parameters
        ----------
        include_parameters : bool, optional
            Whether to include the kernel parameters in the visualization.

        Returns
        -------
        str
            A string containing the visual representation of the tree kernel.
        """

        pre_order_expression, num_nodes, pre_level_map = self.get_pre_order()

        description = tree_visualization(
            pre_order_expression[:num_nodes],
            pre_level_map,
            self.atoms + self.operators,
            self.parameters.value,
            self.is_operator.value,
            self.root_idx.value,
            self.max_depth.value,
            include_parameters=include_parameters,
            print_str=False,
            return_str=True,
        )
        assert isinstance(description, str)
        return description

    def __str__(self) -> str:
        """
        Construct a visual representation of the tree kernel.
        See tree_visualization for more information.

        Returns
        -------
        str
            A string containing the visual representation of the tree kernel (if
            return_str is True).
        """
        return self._tree_viz(include_parameters=True)

    def cross_covariance(
        self,
        x: Float[jnp.ndarray, " M"] | Float[jnp.ndarray, " M 1"],
        y: Float[jnp.ndarray, " N"] | Float[jnp.ndarray, " N 1"],
    ) -> Float[jnp.ndarray, "M N"]:
        """
        Calculate the cross-covariance matrix between x and y
        for the tree kernel.

        Parameters
        ----------
        x : Float[jnp.ndarray, " M"] | Float[jnp.ndarray, " M 1"]
            Input x data.
        y : Float[jnp.ndarray, " N"] | Float[jnp.ndarray, " N 1"]
            Input y data.


        Returns
        -------
        Float[jnp.ndarray, "M N"]
            The cross-covariance matrix.
        """
        return _evaluate_tree(
            x,
            y,
            self.post_order_expression.value,
            self.post_level_map.value,
            self.is_operator.value,
            self.parameters.value,
            tuple([atom.cross_covariance for atom in self.atoms]),
            self.operators,
            (int(len(x)), int(len(y))),
            int(self.num_atoms.value),
            int(self.max_depth.value),
        )

    def gram(
        self,
        x: Float[jnp.ndarray, " M"] | Float[jnp.ndarray, " M 1"],
    ) -> Float[jnp.ndarray, "M M"]:
        """
        Calculate the gram matrix for a given input x using
        the tree kernel.

        Parameters
        ----------
        x : Float[jnp.ndarray, " D"] | Float[jnp.ndarray, " D 1"]
            Input x data.

        Returns
        -------
        Float[jnp.ndarray, "M M"]
            The gram matrix.
        """
        return self.cross_covariance(x, x)

    def _gram_train(
        self,
        x: Float[jnp.ndarray, " M"] | Float[jnp.ndarray, " M 1"],
    ) -> Float[jnp.ndarray, "M M"]:
        """
        Calculate the gram matrix for a given input x using
        the tree kernel. This version has the number of datapoints
        fixed to the initial number of datapoints in the training data,
        and is used for evaluation in a jit-compatible way.

        Returns
        -------
        Float[jnp.ndarray, "M M"]
            The gram matrix.
        """

        return _evaluate_tree(
            x,
            x,
            self.post_order_expression.value,
            self.post_level_map.value,
            self.is_operator.value,
            self.parameters.value,
            tuple([atom.cross_covariance for atom in self.atoms]),
            self.operators,
            (int(self.num_datapoints.value), int(self.num_datapoints.value)),
            int(self.num_atoms.value),
            int(self.max_depth.value),
        )

    def get_pre_order(
        self,
    ) -> tuple[Int[jnp.ndarray, " D"], ScalarInt, Int[jnp.ndarray, " D"]]:
        """
        Get the pre-order traversal expression of the tree, as well as
        the index of the last node in the tree and a map from the pre-order
        index to the level-order index.


        Returns
        -------
        Int[jnp.ndarray, " D"]
            The pre-order traversal expression of the tree.
        ScalarInt
            The total number of nodes in the tree.
        Int[jnp.ndarray, " D"]
            A map from the pre-order index to the level-order index. The value at a
            given index in the pre-order expression corresponds to the index of the
            same node in the level-order expression
        """

        initial_expression = jnp.full(self.max_nodes.value, -1)
        initial_expression_pointer = 0
        pre_level_map = jnp.copy(initial_expression)
        initial_stack = (
            jnp.copy(initial_expression).at[0].set(self.root_idx)
        )  # fill with root
        initial_stack_pointer = 0

        initial_state = (
            initial_expression,
            initial_expression_pointer,
            pre_level_map,
            initial_stack,
            initial_stack_pointer,
        )

        return level_order_to_pre_order(
            self.tree_expression.value,
            initial_state,
            self.max_nodes.value,
        )

    def get_post_order_and_node_metrics(
        self,
    ) -> tuple[
        Int[jnp.ndarray, " D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
    ]:
        """
        Get the post-order traversal expression of the tree, the output is
        - the post-order traversal expression of the tree,
        - the total number of nodes in the tree,
        - a map from the post-order index to the level-order index,
        - an array containing the sizes of the nodes in the tree (located at the
          level-order index),
        - an array containing the heights of the nodes in the tree (located at
          the level-order index).
        - a map from the leaf index to the level-order index,
        - the total number of leaves in the tree.

        (Unlike the get_pre_order method, this method also returns the
        node sizes, heights, leaf level map, and number of leaves.)

        Returns
        -------
        Int[jnp.ndarray, " D"]
            The post-order traversal expression of the tree.
        ScalarInt
            The total number of nodes in the tree.
        Int[jnp.ndarray, " D"]
            A map from the post-order index to the level-order index. The value at a
            given index in the post-order expression corresponds to the index of the
            same node in the level-order expression
        Int[jnp.ndarray, " D"]
            An array containing the sizes of the nodes in the tree. The size of a node
            is the number of nodes in the subtree rooted at that node. The size of a
            given node is the value at its index in the level-order expression.
        Int[jnp.ndarray, " D"]
            An array containing the heights of the nodes in the tree. The height of a
            node is the length of the longest path from the node to a leaf node
            (i.e. the height of a leaf node is 0). The height of a given
            node is the value at its index in the level-order expression.
        Int[jnp.ndarray, " D"]
            A map from the leaf index to the level-order index. The value at a given
            index in the leaf level map corresponds to the index of the node with
            these parameters in the level-order expression.
            NOTE: The leaf index is decided based on the depth of the tree, and might
            not be in a simple order. See
            'gallifrey.utils.tree_helper.get_parameter_leaf_idx' for more information.
        """
        initial_expression = jnp.full(self.max_nodes.value, -1)
        initial_expression_pointer = 0
        post_level_map = jnp.copy(initial_expression)
        initial_stack = (
            jnp.copy(initial_expression).at[0].set(self.root_idx)
        )  # fill with root
        initial_stack_pointer = 0
        last_processed_idx = -1
        node_sizes = jnp.full(self.max_nodes, 0)
        node_heights = jnp.copy(node_sizes)
        leaf_level_map = jnp.full(self.max_leaves, -1)

        initial_state = (
            initial_expression,
            initial_expression_pointer,
            post_level_map,
            initial_stack,
            initial_stack_pointer,
            last_processed_idx,
            node_sizes,
            node_heights,
            leaf_level_map,
        )

        return level_order_to_post_order_and_metrics(
            self.tree_expression.value,
            initial_state,
            self.is_operator.value,
            self.max_nodes.value,
            self.max_depth.value,
        )

    def print_atoms(self) -> None:
        """
        Print the atoms in the atom library, together with their parameter names.
        """
        for atom in self.atoms:
            print(f"{atom.name}: {atom.parameter_names}")

    def display(self) -> None:
        print(self.__str__())


@jit
def level_order_to_pre_order(
    level_order: Int[jnp.ndarray, " D"],
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
    ],
    max_nodes: ScalarInt,
) -> tuple[Int[jnp.ndarray, " D"], ScalarInt, Int[jnp.ndarray, " D"]]:
    """
    Convert a level-order expression of a tree to a pre-order expression.
    Also returns the total number of nodes in the tree. The
    pre-order expression is used to visualize the tree kernel.

    NOTE: This function should be called via the get_pre_order method of the
    TreeKernel.

    Parameters
    ----------
    level_order : Int[jnp.ndarray, " D"]
        The level-order expression of the tree.
    initial_state : tuple
        The initial state, containing the pre-order expression, the
        pre-order pointer, the pre_level_map, the stack, and the stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree expression.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The pre-order expression of the tree.
    ScalarInt
        The total number of nodes in the tree.
    Int[jnp.ndarray, " D"]
        A map from the pre-order index to the level-order index. The value at a given
        index in the pre-order expression corresponds to the index of the same node in
        the level-order expression.

    """

    def traverse_tree(state: tuple) -> tuple:
        """Traverse the tree expression in level-order and convert it to
        pre-order."""
        pre_order, pre_order_pointer, pre_level_map, stack, stack_pointer = state

        # pop the top of the stack
        level_order_idx = stack[stack_pointer]
        stack_pointer = stack_pointer - 1

        # add the current node to the pre-order output
        new_pre_order = pre_order.at[pre_order_pointer].set(
            level_order[level_order_idx]
        )
        new_pre_order_pointer = pre_order_pointer + 1
        # update the pre_level_map
        new_pre_level_map = pre_level_map.at[pre_order_pointer].set(level_order_idx)

        # check if the current node is a leaf
        right_child_idx = get_child_idx(level_order_idx, "l")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            level_order[right_child_idx] == -1,
        )

        def push(stack_and_pointer: tuple) -> tuple:
            """If the node is not a leaf, push the new
            child nodes onto the stack.
            """
            stack, pointer = stack_and_pointer

            left_child = get_child_idx(level_order_idx, "l")
            right_child = get_child_idx(level_order_idx, "r")

            stack_updated = stack.at[pointer + 1].set(right_child)
            stack_updated = stack_updated.at[pointer + 2].set(left_child)

            pointer_updated = pointer + 2
            return stack_updated, pointer_updated

        def dont_push(stack_and_pointer: tuple) -> tuple:
            """If the node is a leaf, do nothing."""
            return stack_and_pointer

        # update the stack based on whether the current node is an operator
        new_stack, new_pointer = lax.cond(
            is_leaf,
            dont_push,
            push,
            operand=(stack, stack_pointer),
        )

        return (
            new_pre_order,
            new_pre_order_pointer,
            new_pre_level_map,
            new_stack,
            new_pointer,
        )

    def condition(state: tuple) -> ScalarBool:
        """Loop condition: Continue until stack is empty."""
        _, _, _, _, stack_pointer = state
        return stack_pointer >= 0

    # iterate over the sample using lax.while_loop
    final_pre_order, num_nodes, final_pre_level_map, _, _ = lax.while_loop(
        condition,
        traverse_tree,
        initial_state,
    )
    return final_pre_order, num_nodes, final_pre_level_map


@jit
def level_order_to_post_order_and_metrics(
    level_order: Int[jnp.ndarray, " D"],
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
        ScalarInt,
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
    ],
    is_operator: Bool[jnp.ndarray, " D"],
    max_nodes: ScalarInt,
    max_depth: ScalarInt,
) -> tuple[
    Int[jnp.ndarray, " D"],
    ScalarInt,
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
]:
    """
    Convert a level-order expression of a tree to a post-order expression, and
    other key metrics of the tree. The post-order expression is used to evaluate the
    tree kernel efficiently.

    Returns:
    - the post-order traversal expression of the tree
    - the number of nodes in the tree
    - a map from the post-order index to the level-order index
    - an array containing the sizes of the nodes in the tree
    - an array containing the heights of the nodes in the tree
    - a map from the leaf index to the level-order index


    NOTE: This function should be called via the get_post_order method of the
    TreeKernel.

    Parameters
    ----------
    level_order : Int[jnp.ndarray, " D"]
        The level-order expression of the tree.
    initial_state : tuple
        The initial state, containing allocated arrays for:
        - the post-order expression
        - the post-order pointer
        - the post_level_map
        - the stack
        - the stack pointer
        - the last processed index
        - the node sizes
        - the node heights
        - the leaf level map
    is_operator : Bool[jnp.ndarray, " D"]
        A boolean array indicating whether the node in the tree expression is an
        operator or not.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree expression.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The post-order expression of the tree.
    ScalarInt
        The index of the last non-empty node in the tree (in the post-order expression).
    Int[jnp.ndarray, " D"]
        A map from the post-order index to the level-order index. The value at a given
        index in the post-order expression corresponds to the index of the same node in
        the level-order expression.
    Int[jnp.ndarray, " D"]
        An array containing the sizes of the nodes in the tree. The size of a node is
        the number of nodes in the subtree rooted at that node. The size of a given node
        is the value at its index in the level-order expression.
    Int[jnp.ndarray, " D"]
        An array containing the heights of the nodes in the tree. The height of a node
        is the length of the longest path from the node to a leaf node (i.e. the height
        of a leaf node is 0). The height of a given node is the value at its index in
        the level-order expression.
    Int[jnp.ndarray, " D"]
        A map from the leaf index to the level-order index. The value at a given index
        in the leaf level map corresponds to the index of the node with these parameters
        in the level-order expression.
        NOTE: The leaf index is decided based on the depth of the tree, and might not be
        in a simple order. See 'gallifrey.utils.tree_helper.get_parameter_leaf_idx' for
        more information.

    """

    def traverse_tree(state: tuple) -> tuple:
        """Traverse the tree expression in level-order and convert it to
        pre-order."""

        (
            _,
            _,
            _,
            stack,
            stack_pointer,
            last_processed_idx,
            _,
            _,
            _,
        ) = state

        # pop the top of the stack
        level_order_idx = stack[stack_pointer]

        # check if the current node is a leaf
        right_child_idx = get_child_idx(level_order_idx, "r")
        is_leaf = jnp.logical_or(
            jnp.logical_or(
                right_child_idx > max_nodes - 1,
                level_order[right_child_idx - 1] == -1,
            ),
            right_child_idx == last_processed_idx,
        )

        def process_non_leaf(state: tuple) -> tuple:
            """If the node is not a leaf, push the node back onto the stack
            and push the children onto the stack.
            """

            (
                post_order,
                post_order_pointer,
                post_level_map,
                stack,
                stack_pointer,
                last_processed_idx,
                node_sizes,
                node_heights,
                leaf_level_map,
            ) = state

            left_child = get_child_idx(level_order_idx, "l")
            right_child = get_child_idx(level_order_idx, "r")

            # Push current node, then right child, then left child onto stack
            stack_updated = stack.at[stack_pointer].set(level_order_idx)
            stack_updated = stack_updated.at[stack_pointer + 1].set(right_child)
            stack_updated = stack_updated.at[stack_pointer + 2].set(left_child)
            stack_pointer_updated = stack_pointer + 2

            return (
                post_order,
                post_order_pointer,
                post_level_map,
                stack_updated,
                stack_pointer_updated,
                last_processed_idx,
                node_sizes,
                node_heights,
                leaf_level_map,
            )

        def process_leaf(state: tuple) -> tuple:
            """If the node is a leaf, add the node to the post-order output."""
            (
                post_order,
                post_order_pointer,
                post_level_map,
                stack,
                stack_pointer,
                _,
                node_sizes,
                node_heights,
                leaf_level_map,
            ) = state

            # add the current node to the post-order output
            post_order_updated = post_order.at[post_order_pointer].set(
                level_order[level_order_idx]
            )
            post_order_pointer_updated = post_order_pointer + 1

            # update the post_level_map
            post_level_map_updated = post_level_map.at[post_order_pointer].set(
                level_order_idx
            )

            # update stack pointer
            stack_pointer_updated = stack_pointer - 1

            # update last processed index
            last_processed_idx_updated = level_order_idx

            # calculate and store the node size and heights
            left_child_idx = get_child_idx(level_order_idx, "l")
            right_child_idx = get_child_idx(level_order_idx, "r")

            left_child_size, left_child_height, right_child_size, right_child_height = (
                lax.cond(
                    right_child_idx < max_nodes,
                    lambda: (
                        node_sizes[left_child_idx],
                        node_heights[left_child_idx],
                        node_sizes[right_child_idx],
                        node_heights[right_child_idx],
                    ),
                    lambda: (0, 0, 0, 0),
                )
            )

            bigger_height = jnp.where(
                left_child_height > right_child_height,
                left_child_height,
                right_child_height,
            )

            node_sizes_updated = node_sizes.at[level_order_idx].set(
                1 + left_child_size + right_child_size
            )
            node_heights_updated = node_heights.at[level_order_idx].set(
                1 + bigger_height
            )

            # update leaf level map if the current node is a kernel atom,
            # not an operator (this differs from being a leaf node, since
            # in scaffolds, operators can be leaf nodes)
            leaf_level_map_updated = jnp.where(
                is_operator[level_order[level_order_idx]],
                leaf_level_map,
                leaf_level_map.at[
                    get_parameter_leaf_idx(
                        level_order_idx,
                        max_depth,
                    )
                ].set(level_order_idx),
            )

            return (
                post_order_updated,
                post_order_pointer_updated,
                post_level_map_updated,
                stack,
                stack_pointer_updated,
                last_processed_idx_updated,
                node_sizes_updated,
                node_heights_updated,
                leaf_level_map_updated,
            )

        # update the stack based on whether the current node is an operator
        new_state = lax.cond(
            is_leaf,
            process_leaf,
            process_non_leaf,
            operand=state,
        )

        return new_state

    def condition(state: tuple) -> ScalarBool:
        """Loop condition: Continue until stack is empty."""
        _, _, _, _, stack_pointer, _, _, _, _ = state
        return stack_pointer >= 0

    # iterate using lax.while_loop
    (
        final_pre_order,
        num_nodes,
        final_post_level_map,
        _,
        _,
        _,
        final_node_sizes,
        final_node_heights,
        final_leaf_level_map,
    ) = lax.while_loop(
        condition,
        traverse_tree,
        initial_state,
    )

    # to get the longest path from the root to a leaf, subtract 1 from the height,
    # since we counted to total number of nodes from the root to the leaf, not the
    # number of edges
    final_node_heights -= 1

    return (
        final_pre_order,
        num_nodes,
        final_post_level_map,
        final_node_sizes,
        final_node_heights,
        final_leaf_level_map,
    )


@partial(
    jit,
    static_argnames=(
        "atoms",
        "operators",
        "data_shape",
        "num_atoms",
        "max_depth",
    ),
)
def _evaluate_tree(
    x: Float[jnp.ndarray, " "] | Float[jnp.ndarray, " D"] | Float[jnp.ndarray, "D 1"],
    y: Float[jnp.ndarray, " "] | Float[jnp.ndarray, " D"] | Float[jnp.ndarray, "D 1"],
    post_order_expression: Int[jnp.ndarray, " D"],
    post_level_map: Int[jnp.ndarray, " D"],
    is_operator: Bool[jnp.ndarray, " D"],
    parameters: Float[jnp.ndarray, " M N"],
    atoms: tuple[AbstractAtom | tp.Callable, ...],
    operators: tuple[AbstractOperator, ...],
    data_shape: tuple[int, int],
    num_atoms: int,
    max_depth: int,
) -> Float[jnp.ndarray, "..."]:
    """
    Evaluate a tree expression by traversing the tree in post-order
    and applying the kernel atom functions and operators.

    NOTE: We are using Equinox's bounded_while_loop to evaluate the tree
    expression. More importantly, we are using its buffers feature to
    store the stack in the loop. This is a workaround to some weird JAX
    behaviour with inplace updates in while loops.
    (THIS IS ACTUALLY NOT CURRENTLY USED ANYMORE, BUT WE LEAVE THIS HERE FOR
    REFERENCE)
    For more information, see
    https://github.com/jax-ml/jax/issues/10197#issuecomment-1621416280
    https://github.com/jax-ml/jax/issues/17640
    https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/loop.py


    Parameters
    ----------
    x : Float[jnp.ndarray, " "] | Float[jnp.ndarray, " D"] | Float[jnp.ndarray, "D 1"]
        Input x data (0D or 1D array of shape (D, ) filled with floats).
    y : Float[jnp.ndarray, " "] | Float[jnp.ndarray, " D"] | Float[jnp.ndarray, "D 1"]
        Input y data (0D or 1D array of shape (D, ) filled with floats).
    post_order_expression : Int[jnp.ndarray, " D"]
        The post-order expression of the tree.
    post_level_map : Int[jnp.ndarray, " D"]
        The map from post order index to level order index.
    is_operator : Bool[jnp.ndarray, " D"]
        A boolean array indicating whether a item in the atom library is an operator.
    parameters : Float[jnp.ndarray, " M N"]
        A jnp array containing the parameters of the kernel functions in the tree
        kernel. The shape of the array is (M, N), where M is the number of nodes in
        the tree and N is the maximum number of parameters of the kernel functions.
    atoms : tuple[AbstractAtom, ...] | tuple[tp.Callable, ...]
        A tuple of kernel atom functions.
    operators : tuple[AbstractOperator, ...]
        A tuple of kernel operators.
    data_shape : tuple[int, int]
        The shape of the input data (len(x), len(y)).
    num_atoms : int
        The number of atoms in the atom library.
    max_depth : int
        The maximum depth of the tree.

    Returns
    -------
    Float[jnp.ndarray, "..."]
        The result of evaluating the tree expression.
    """

    def evaluate(state: tuple) -> tuple:
        """Evaluate the tree expression at a given node index."""
        current_idx, *_ = state

        tree_level_idx = post_level_map[current_idx]
        node_value = post_order_expression[current_idx]
        is_op = is_operator[node_value]

        def eval_leaf_node(state: tuple) -> tuple:
            """Evaluate a leaf node."""
            current_idx, stack, pointer = state
            kernel_evaluation = lax.switch(
                node_value,
                atoms,
                x,
                y,
                parameters[get_parameter_leaf_idx(tree_level_idx, max_depth)],
            )
            new_stack = stack.at[pointer].set(kernel_evaluation)
            return (current_idx + 1, new_stack, pointer + 1)

        def eval_operator_node(state: tuple) -> tuple:
            """Evaluate an operator node."""
            _, stack, pointer = state
            left_child = jnp.array([stack[pointer - 1]])
            right_child = jnp.array([stack[pointer - 2]])
            kernel_evaluation = lax.switch(
                node_value - num_atoms,
                operators,
                left_child,
                right_child,
            )
            new_stack = stack.at[pointer - 2].set(kernel_evaluation)
            return (current_idx + 1, new_stack, pointer - 1)

        new_state = lax.cond(
            is_op,
            eval_operator_node,
            eval_leaf_node,
            operand=state,
        )
        return new_state

    def condition(state: tuple) -> Bool[jnp.ndarray, " "]:
        """Loop condition: Continue until the end of the tree expression (by
        encountering empty node, which is given by -1 value in
        post_order_expression).
        """
        current_idx, *_ = state
        return post_order_expression[current_idx] >= 0

    # def buffers(state: tuple) -> tp.Any:
    #     """Turn stack into Equinox buffer."""
    #     return state[1]

    # create initital state
    initial_stack = jnp.zeros(
        [
            calculate_max_stack_size(max_depth),
            data_shape[0],
            data_shape[1],
        ]
    )
    initial_pointer = 0
    initial_index = 0
    initital_state = (initial_index, initial_stack, initial_pointer)

    _, final_stack, *_ = bounded_while_loop(
        condition,
        evaluate,
        initital_state,
        max_steps=int(calculate_max_nodes(max_depth)),
        kind="checkpointed",
        # buffers=buffers,
        checkpoints=5,  # using checkpoint equation in equinox and assuming,
        # kernels with only ever have up to ~20 nodes,
        # might need to be adjusted for larger trees
    )
    return final_stack[0]


def tree_visualization(
    pre_order_expression: Int[jnp.ndarray, " D"],
    pre_level_map: Int[jnp.ndarray, " D"],
    atom_library: tuple[AbstractAtom | AbstractOperator, ...],
    parameters: Float[jnp.ndarray, " M N"],
    is_operator: Bool[jnp.ndarray, " D"],
    root_idx: Int[jnp.ndarray, ""],
    max_depth: ScalarInt,
    include_parameters: bool = True,
    print_str: bool = True,
    return_str: bool = False,
) -> None | str:
    """
    Construct a visual representation of a tree expression.

    Parameters
    ----------
    pre_order_expression : Int[jnp.ndarray, " D"]
        The tree expression in pre-order traversal notation.
    pre_level_map : Int[jnp.ndarray, " D"]
        A map from the pre-order index to the level-order index. The value at a given
        index in the pre-order expression corresponds to the index of the same node in
        the level-order expression.
    atom_library : tuple[AbstractAtom | AbstractOperator, ...]
        A tuple of kernel atom functions and operators to evaluate the tree expression.
    parameters : Float[jnp.ndarray, " M N"]
        The parameters of the kernel functions in the tree kernel.
    is_operator : Bool[jnp.ndarray, " D"]
        A boolean array indicating whether a item in the atom library is an operator.
    root_idx : Int[jnp.ndarray, ""]
        The index of the root node in the tree expression.
    max_depth : ScalarInt
        The maximum depth of the tree.
    include_parameters : bool, optional
        Whether to include the kernel parameters in the string, by default True.
    print_str : bool, optional
        Whether to print the visual representation of the tree, by default True.
    return_str : bool, optional
        Whether to return the visual representation as a string, by default False

    Returns
    -------
    str
        A string containing the visual representation of the tree expression (if
        return_str is True).
    """
    descr_str = ""

    for pre_order_idx, node_value in enumerate(pre_order_expression):
        tree_level_idx = pre_level_map[pre_order_idx]
        atom = atom_library[node_value]

        depth = get_depth(tree_level_idx) - get_depth(root_idx)
        prefix = "    " * (depth - 1) + "└── " if depth > 0 else ""

        node_representation = f"{atom.name}"

        if not is_operator[node_value] and include_parameters:
            node_parameters = parameters[
                get_parameter_leaf_idx(tree_level_idx, max_depth)
            ]
            parameter = node_parameters[: atom.num_parameter]
            node_representation += f": {parameter}"

        descr_str += f"{prefix}{node_representation}\n"

    if print_str:
        print(descr_str)
    if return_str:
        return descr_str
    return None
