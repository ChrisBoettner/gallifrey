from functools import partial

from flax import nnx
from jax import debug, jit, lax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Int, PRNGKeyArray

from gallifrey.kernels.prior import KernelPrior, TreeStructurePrior
from gallifrey.utils.tree_helper import get_child_idx, pick_random_node
from gallifrey.utils.typing import ScalarBool, ScalarInt


@partial(jit, static_argnames=("kernel_prior", "verbosity"))
def subtree_replace_move(
    key: PRNGKeyArray,
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    verbosity: int = 0,
) -> nnx.State:
    """
    Perform a subtree replacement move on a kernel state.

    In this move, a subtree is replaced with a new
    subtree sampled from the prior. The root of the subtree
    to be replaced is chosen uniformly at random.

    The parameter from the old tree are copied to the new tree,
    while the parameters for the new subtree are sampled from the prior.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_state : nnx.State
        The original kernel state, must have attributes:
        - tree_expression
        - post_level_map
    kernel_prior : KernelPrior
        The prior to sample the new subtree from.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.


    Returns
    -------
    nnx.State
        The updated kernel state after the move.

    """
    key, structure_key, parameter_key = jr.split(key, 3)

    # perform the subtree replacement move on the structure
    new_structure, changes, _ = structure_subtree_replace_move(
        structure_key,
        kernel_state.tree_expression.value,  # type: ignore
        kernel_state.post_level_map.value,  # type: ignore
        kernel_prior.kernel_structure_prior,
        verbosity=verbosity,
    )

    # create a new kernel with the updated structure
    new_kernel = nnx.merge(kernel_prior.graphdef, kernel_state)
    new_kernel.init_tree(
        new_structure,
        kernel_prior.kernel_library,
        kernel_prior.max_depth,
        kernel_prior.num_datapoints,
    )

    # sample the new parameters, while keeping the old ones
    _, blank_new_state = nnx.split(new_kernel)
    new_state, _ = kernel_prior.parameter_prior.sample_subset(
        parameter_key,
        blank_new_state,
        considered_nodes=changes,
    )
    return new_state


def structure_subtree_replace_move(
    key: PRNGKeyArray,
    tree_expression: Int[jnp.ndarray, " D"],
    post_level_map: Int[jnp.ndarray, " D"],
    tree_prior: TreeStructurePrior,
    verbosity: int = 0,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    ScalarInt,
]:
    """
    Perform a subtree replacement move on the kernel structure array.
    In this move, a subtree is replaced with a new
    subtree sampled from the prior. The root of the subtree
    to be replaced is chosen uniformly at random.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression (level order).
    post_level_map : Int[jnp.ndarray, " D"]
        The array that maps the post order index to the level order index.
        (This is useful, since the post_level_map contains all the indices
        of the nodes in the tree).
    tree_prior : TreeStructurePrior
        The prior to sample the new subtree from.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression after the move.
    Int[jnp.ndarray, " D"]
        An array containing the indices of the nodes that were changed.
    ScalarInt
        The index where the new subtree was attached.

    """
    key, idx_key, subtree_key = jr.split(key, 3)

    # sample a node to replace uniformly from non-empty nodes
    replace_idx = pick_random_node(idx_key, post_level_map)

    # if tree_expression[replace_idx] == empty_node_value:
    #     raise RuntimeError(
    #         "The chosen index is that of an empty node, "
    #         "that should not be possible. Check inputs."
    #     )

    if verbosity > 2:
        debug.print("Replacing node at index {}", replace_idx)

    updated_tree_expression, changed_nodes = replace_subtree_structure(
        subtree_key,
        tree_expression,
        replace_idx,
        tree_prior,
        verbosity=verbosity,
    )
    return updated_tree_expression, changed_nodes, replace_idx


def replace_subtree_structure(
    key: PRNGKeyArray,
    tree_expression: Int[jnp.ndarray, " D"],
    replace_idx: ScalarInt,
    tree_prior: TreeStructurePrior,
    verbosity: int = 0,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
]:
    """
    Replace a subtree in the tree expression with a
    new subtree sampled from the prior.

    In contrast to the subtree_replace_move function,
    this function uses a fixed index for
    the root of the subtree to replace.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression (level order).
    replace_idx : ScalarInt
        The index of the node where to attach the new subtree.
    tree_prior : TreeStructurePrior
        The prior to sample the new subtree from.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression.
    Int[jnp.ndarray, " D"]
        An array containing the indices of the nodes that were changed.
    """

    new_subtree = tree_prior.sample_single(
        key,
        root_idx=replace_idx,
    )

    if verbosity > 2:
        debug.print("New subtree: {}", new_subtree)

    return replace_subtree_structure_with(
        tree_expression,
        new_subtree,
        replace_idx,
    )


def replace_subtree_structure_with(
    tree_expression: Int[jnp.ndarray, " D"],
    subtree_expression: Int[jnp.ndarray, " D"],
    replace_idx: ScalarInt,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
]:
    """
    Replace a subtree in the tree_expression with
    another subtree. The subtree to replace is specified
    by the replace_idx.

    In contrast to the subtree_replace_move and replace_subtree functions,
    this function takes the tree_expression and subtree_expression as inputs.
    No random key is required.

    Parameters
    ----------
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression (level order).
    subtree_expression : Int[jnp.ndarray, " D"]
        The subtree expression (level order). The array must
        have the same length as the tree_expression.
    replace_idx : ScalarInt
        The index of the node where to attach the new subtree.


    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression, with the subtree replaced.
    Int[jnp.ndarray, " D"]
        An array containing the indices of the nodes that were changed.

    """

    max_nodes = tree_expression.size

    # if max_nodes != subtree_expression.size:
    #     raise ValueError(
    #         "tree_expression and subtree_expression must have the same length."
    #     )

    # create initial state
    changed_values = jnp.full(max_nodes, -1, dtype=tree_expression.dtype)
    changed_pointer = 0
    stack = jnp.copy(tree_expression).at[0].set(replace_idx)
    stack_pointer = 0
    initial_state = (
        tree_expression,
        subtree_expression,
        changed_values,
        changed_pointer,
        stack,
        stack_pointer,
    )

    new_tree_expression, changed_values, _ = _replace_subtree_structure_with(
        initial_state,
        max_nodes,
    )

    return new_tree_expression, changed_values


@jit
def _replace_subtree_structure_with(
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        ScalarInt,
    ],
    max_nodes: ScalarInt,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    ScalarInt,
]:
    """
    Replace a subtree in the tree expression with another subtree.

    NOTE: This function is not meant to be called directly. Use replace_subtree_with
    instead.

    Parameters
    ----------
    initial_state : tuple
        The initial state of the traversal. Contains the tree expression,
        the subtree expression, the changed values, the changed pointer,
        the stack and the stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression with the subtree replaced.
    Int[jnp.ndarray, " D"]
        An array containing the indices of the nodes that were changed (same length
        as tree_expression).
    ScalarInt
        The number of nodes that were changed.


    """

    def traverse(state: tuple) -> tuple:
        """Traverse the tree expression in level-order."""
        (
            tree_expression,
            subtree_expression,
            changed_values,
            changed_pointer,
            stack,
            pointer,
        ) = state

        idx = stack[pointer]

        # check if the current node is a leaf in either tree
        right_child_idx = get_child_idx(idx, "r")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            jnp.logical_and(
                tree_expression[right_child_idx] == -1,
                subtree_expression[right_child_idx] == -1,
            ),
        )

        def process_leaf(stack_and_pointer: tuple) -> tuple:
            """If the current node is a leaf, move parent."""
            stack, pointer = stack_and_pointer
            return stack, pointer - 1

        def process_non_leaf(stack_and_pointer: tuple) -> tuple:
            """If the current node is not a leaf, update the stack."""
            stack, pointer = stack_and_pointer

            left_child_idx = get_child_idx(idx, "l")
            right_child_idx = get_child_idx(idx, "r")

            new_stack = stack.at[pointer].set(left_child_idx)
            new_stack = new_stack.at[pointer + 1].set(right_child_idx)

            return new_stack, pointer + 1

        # update the stack and pointer
        new_stack, new_pointer = lax.cond(
            is_leaf,
            process_leaf,
            process_non_leaf,
            operand=(stack, pointer),
        )

        # override the current node in the tree expression with subtree node
        old_value = tree_expression[idx]
        new_value = subtree_expression[idx]
        tree_expression_updated = tree_expression.at[idx].set(new_value)

        # if values are different, update the changed values and pointer
        changed_values_updated, changed_pointer_updated = lax.cond(
            old_value != new_value,
            lambda x: (x[0].at[x[1]].set(idx), x[1] + 1),
            lambda x: x,
            operand=(changed_values, changed_pointer),
        )

        return (
            tree_expression_updated,
            subtree_expression,
            changed_values_updated,
            changed_pointer_updated,
            new_stack,
            new_pointer,
        )

    def condition(state: tuple) -> ScalarBool:
        """Check if the traversal is complete."""
        _, _, _, _, _, pointer = state
        return pointer >= 0

    # iterate over the sample using lax.while_loop
    new_tree_expression, _, changed_values, changed_pointer, _, _ = lax.while_loop(
        condition,
        traverse,
        initial_state,
    )

    return new_tree_expression, changed_values, changed_pointer
