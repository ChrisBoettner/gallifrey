import beartype.typing as tp
import jax.numpy as jnp
from beartype.typing import Literal
from jax import jit, lax
from jax import random as jr
from jaxtyping import Int, PRNGKeyArray

from gallifrey.utils.typing import ScalarBool, ScalarFloat, ScalarInt


def get_child_idx(idx: ScalarInt, direction: Literal["l", "r"]) -> ScalarInt:
    """
    Get the index of the left or right child of a node in a binary tree,
    given by
    2 * idx + 1 for left child
    2 * idx + 2 for right child.

    Compare: https://en.wikipedia.org/wiki/Binary_tree#Arrays

    Parameters
    ----------
    idx : ScalarInt
        The index of the node.
    direction : Literal["l", "r"]
        The side of the child to get. "l" for left child, "r" for right child.

    Returns
    -------
    ScalarInt
        The index of the child.
    """
    direction_map = {"l": 1, "r": 2}
    return 2 * idx + direction_map[direction]


def get_parent_idx(idx: ScalarInt) -> ScalarInt:
    """
    Get the index of the parent of a node in a binary tree,
    given by floor((idx - 1) / 2).

    Compare: https://en.wikipedia.org/wiki/Binary_tree#Arrays

    Parameters
    ----------
    idx : ScalarInt
        The index of the node.

    Returns
    -------
    int
        The index of the parent.
    """
    return (idx - 1) // 2


def get_depth(idx: ScalarInt, tolerance: ScalarFloat = 1e-10) -> ScalarInt:
    """
    Get the depth of a node in a binary tree.

    Parameters
    ----------
    idx : ScalarInt
        The index of the node.
    tolerance : ScalarFloat, optional
        On occasion, floating point errors may lead to a depth that is off by one.
        This can happen whenever the index is a power of 2. To avoid this, a tolerance
        is added to the index before taking the logarithm. By default 1e-10.

    Returns
    -------
    int
        The depth of the node.
    """
    return jnp.log2(idx + 1 + tolerance).astype(int)


def calculate_max_nodes(max_depth: ScalarInt) -> ScalarInt:
    """
    Calculate the maximum number of nodes in a binary tree based
    on the maximum depth of the tree.
     Since each node in the tree can have at most two children,
    the maximum number of nodes in the tree is given by
    2^(max_depth + 1) - 1.

    Parameters
    ----------
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.

    Returns
    -------
    ScalarInt
        The maximum number of nodes in the tree that describes the kernel structure.
    """
    return 2 ** (max_depth + 1) - 1


def calculate_max_stack_size(max_depth: ScalarInt) -> ScalarInt:
    """
    For a given tree depth, calculate the maximum size of the stack
    needed to traverse the tree in level order.

    Parameters
    ----------
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.

    Returns
    -------
    ScalarInt
        The maximum size of the stack needed to traverse the tree.
    """
    return max_depth + 1


def calculate_max_leaves(max_depth: ScalarInt) -> ScalarInt:
    """
    Calculate the maximum number of leaves in a binary tree based
    on the maximum depth of the tree.
    This corresponds to the number of nodes at the maximum depth
    of the tree, which is given by 2^max_depth.

    Parameters
    ----------
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.

    Returns
    -------
    ScalarInt
        The maximum number of leaves in the tree that describes the
        kernel structure.
    """
    return 2**max_depth


def calculate_max_depth(
    max_nodes: ScalarInt,
    tolerance: ScalarFloat = 1e-10,
) -> ScalarInt:
    """
    Calculate the maximum depth of the tree that describes the kernel structure,
    given the maximum number of nodes in the tree.

    Parameters
    ----------
    max_nodes : ScalarInt
        The maximum number of nodes in the tree that describes the kernel structure.
    tolerance : ScalarFloat, optional
        On occasion, floating point errors may lead to a depth that is off by one.
        This can happen whenever the index is a power of 2. To avoid this, a tolerance
        is added to the index before taking the logarithm. By default 1e-10.

    Returns
    -------
    ScalarInt
        The maximum depth of the nested kernel structure tree.
    """
    return get_depth(max_nodes - 1, tolerance)  # idx of last node is max_nodes - 1


def get_leftmost_leaf(idx: ScalarInt, max_depth: ScalarInt) -> ScalarInt:
    """
    For a tree with a given maximum depth, get the index of the leftmost child
    leaf at the maximum depth from a given index (in level order notation).

    Since the equation for a left child is 2 * idx + 1, the left child of the left
    child is 4*i+3, and in general the left-most child at max depth d is
    2^(max_depth-current_depth)*idx + 2^(max_depth-current_depth)-1 =
    2^(max_depth-current_depth)*(idx+1) - 1., which
    can be efficiently implemented using bit-shift operations.


    Parameters
    ----------
    idx : _type_
        _description_
    max_depth : _type_
        _description_

    Returns
    -------
    ScalarInt
        The index of the leftmost leaf at the maximum depth.
    """
    current_depth = get_depth(idx)
    diff = max_depth - current_depth
    return jnp.left_shift(idx + 1, diff) - 1


def get_parameter_leaf_idx(
    tree_level_idx: ScalarInt,
    max_depth: ScalarInt,
) -> ScalarInt:
    """
    Get the index in the parameter array for a given node in the tree expression.

    We work with the convention that, if a leaf node is not at the maximum depth,
    the parameters are stored in the leftmost leaf at the maximum depth. Substracting
    the index of the very left most leaf (from the root) therefore gives a unique index
    for each possible leaf node in the tree.

    Parameters
    ----------
    tree_level_idx : ScalarInt
        The index of the node in the tree expression (level order).
    max_depth : ScalarInt
        The maximum depth of the tree.

    Returns
    -------
    ScalarInt
        The index of the parameters in the parameter array for a given node.

    """
    return get_leftmost_leaf(tree_level_idx, max_depth) - get_leftmost_leaf(
        0, max_depth
    )


def clear_subtree(
    tree_expression: Int[jnp.ndarray, " D"],
    clear_idx: ScalarInt,
    fill_value: ScalarFloat = -1,
) -> Int[jnp.ndarray, " D"]:
    """
    Clear a subtree in a level order tree expression.

    NOTE: The new tree expression will not be a valid
    kernel expression, as the subtree removal leaves
    a hole in the expression.

    Parameters
    ----------
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, given in level order notation.
    clear_idx : ScalarInt
        The index of the root node of the subtree to clear.
    fill_value : ScalarFloat, optional
        The value to fill the cleared nodes with. By default
        -1.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression with the subtree cleared.

    """
    max_nodes = tree_expression.size

    # create initial state
    stack = jnp.zeros(max_nodes, dtype=tree_expression.dtype)
    stack = stack.at[0].set(clear_idx)
    stack_pointer = 0
    initial_state = (tree_expression, stack, stack_pointer)

    return _clear_subtree(
        initial_state,
        max_nodes,
        fill_value,
    )


@jit
def _clear_subtree(
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
    ],
    max_nodes: ScalarInt,
    fill_value: ScalarInt,
) -> Int[jnp.ndarray, " D"]:
    """
    Clear a subtree in a level order tree expression.

    NOTE: This function is not meant to be called directly. Use clear_subtree
    instead.

    Parameters
    ----------
    initial_state : tuple
        The initial state of the traversal. Contains the tree expression,
        the stack and the stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree.
    fill_value : ScalarInt
        The value to fill the cleared nodes with.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The updated tree expression with the subtree cleared.

    """

    def traverse(state: tuple) -> tuple:
        """Traverse the tree expression in level-order."""
        tree_expression, stack, pointer = state

        # clear the current node
        idx = stack[pointer]
        tree_expression_updated = tree_expression.at[idx].set(fill_value)

        # check if the current node is a leaf
        right_child_idx = get_child_idx(idx, "r")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            tree_expression[right_child_idx - 1] == -1,
        )

        def process_leaf(stack_and_pointer: tuple) -> tuple:
            """If the current node is a leaf, move to the parent node
            (node has already been cleared)."""
            stack, pointer = stack_and_pointer
            return stack, pointer - 1

        def process_non_leaf(stack_and_pointer: tuple) -> tuple:
            """If the current node is not a leaf, update the stack
            (add the children of the current node)."""
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

        return tree_expression_updated, new_stack, new_pointer

    def condition(state: tuple) -> ScalarBool:
        """Check if the traversal is complete."""
        _, _, pointer = state
        return pointer >= 0

    # iterate over the sample using lax.while_loop
    new_tree_expression, _, _ = lax.while_loop(
        condition,
        traverse,
        initial_state,
    )
    return new_tree_expression


def get_subtree(
    tree_expression: Int[jnp.ndarray, " D"],
    subtree_root_idx: ScalarInt,
    new_root_idx: tp.Optional[ScalarInt] = None,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, "2 D"],
]:
    """
    Retrieve a subtree from a level order tree expression, starting
    at a given attachment node.

    The tree can be re-rooted at a different node, if wanted.

    Parameters
    ----------
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, given in level order notation.
    subtree_root_idx : ScalarInt
        The index of node where the subtree is rooted.
    new_root_idx : tp.Optional[ScalarInt], optional
        The new index of the root node, if the subtree is to be re-rooted,
        by default None. If None, the subtree is returned in its original
        location.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The (level order) tree expression of the subtree, same length as the
        input tree expression.
    Int[jnp.ndarray, "2 D"]
        The indices of the nodes in the subtree. Two rows, the first row
        contains the indices in the original tree, the second row contains
        the indices in the subtree. (Will be the same if the tree is not
        re-rooted.)

    """
    max_nodes = tree_expression.size

    # if tree_expression[subtree_root_idx] == -1:
    #     raise ValueError("The start index must be a non-empty node.")

    new_root_idx = subtree_root_idx if new_root_idx is None else new_root_idx

    # create initial state
    new_tree_expression = jnp.full(max_nodes, -1)
    index_array = jnp.full((2, max_nodes), -1)
    index_pointer = 0
    stack = jnp.copy(new_tree_expression).at[0].set(subtree_root_idx)
    new_stack = jnp.copy(stack).at[0].set(new_root_idx)
    stack_pointer = 0

    initial_state = (
        new_tree_expression,
        index_array,
        index_pointer,
        stack,
        new_stack,
        stack_pointer,
    )

    subtree_expression, index_array, _ = _get_subtree(
        tree_expression,
        initial_state,
        max_nodes,
    )

    index_array = index_array
    # if jnp.any(index_array >= max_nodes):
    #     raise ValueError(
    #         "The index array is out of bounds. This could happen if the tree "
    #         "is re-rooted to a spot where one of its leaves exceeds the "
    #         "maximum number of nodes."
    #     )
    return subtree_expression, index_array


@jit
def _get_subtree(
    tree_expression: Int[jnp.ndarray, " D"],
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, "2 D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
    ],
    max_nodes: ScalarInt,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, "2 D"],
    ScalarInt,
]:
    """
    Traverse the tree expression in level-order, starting at a given node
    and retrieve the subtree rooted at that node. Perform a re-rooting if
    wanted.

    NOTE: This function is not meant to be called directly. Use get_subtree
    instead.

    Parameters
    ----------
    initial_state : tuple
        The initial state of the traversal. Contains the new tree expression,
        the new root index, the index array, the index pointer, the stack, the
        stack pointer, the new stack, and the new stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The (level order) tree expression of the subtree, same length as the
        input tree expression.
    Int[jnp.ndarray, "2 D"]
        The indices of the nodes in the subtree. Two rows, the first row
        contains the indices in the original tree, the second row contains
        the indices in the subtree. (Will be the same if the tree is not
        re-rooted.)

    """

    def traverse(state: tuple) -> tuple:
        """Traverse the tree expression in level-order."""
        (
            subtree_expression,
            index_array,
            index_pointer,
            old_stack,
            new_stack,
            stack_pointer,
        ) = state

        # get old and new indices
        old_idx = old_stack[stack_pointer]
        new_idx = new_stack[stack_pointer]

        # fill index array
        index_array_updated = index_array.at[0, index_pointer].set(old_idx)
        index_array_updated = index_array_updated.at[1, index_pointer].set(new_idx)
        index_pointer_updated = index_pointer + 1

        # fill the new tree expression with the current node (at the new index)
        subtree_expression_updated = subtree_expression.at[new_idx].set(
            tree_expression[old_idx]
        )

        # check if the current node is a leaf
        right_child_idx = get_child_idx(old_idx, "r")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            tree_expression[right_child_idx - 1] == -1,
        )

        def process_leaf(stacks_and_pointer: tuple) -> tuple:
            """If the current node is a leaf, move to the parent node."""
            old_stack, new_stack, pointer = stacks_and_pointer
            return old_stack, new_stack, pointer - 1

        def process_non_leaf(stacks_and_pointer: tuple) -> tuple:
            """If the current node is not a leaf, update the stacks
            (add the children of the current node)."""
            old_stack, new_stack, pointer = stacks_and_pointer

            left_child_idx = get_child_idx(old_idx, "l")
            right_child_idx = get_child_idx(old_idx, "r")

            new_left_child_idx = get_child_idx(new_idx, "l")
            new_right_child_idx = get_child_idx(new_idx, "r")

            old_stack_updated = old_stack.at[pointer].set(left_child_idx)
            old_stack_updated = old_stack_updated.at[pointer + 1].set(right_child_idx)

            new_stack_updated = new_stack.at[pointer].set(new_left_child_idx)
            new_stack_updated = new_stack_updated.at[pointer + 1].set(
                new_right_child_idx
            )

            return old_stack_updated, new_stack_updated, pointer + 1

        # update the stack and pointer
        old_stack_updated, new_stack_updated, pointer_updated = lax.cond(
            is_leaf,
            process_leaf,
            process_non_leaf,
            operand=(old_stack, new_stack, stack_pointer),
        )

        return (
            subtree_expression_updated,
            index_array_updated,
            index_pointer_updated,
            old_stack_updated,
            new_stack_updated,
            pointer_updated,
        )

    def condition(state: tuple) -> ScalarBool:
        """Check if the traversal is complete."""
        _, _, _, _, _, stack_pointer = state
        return stack_pointer >= 0

    # iterate over the sample using lax.while_loop
    final_subtree_expression, final_index_array, final_index_pointer, _, _, _ = (
        lax.while_loop(
            condition,
            traverse,
            initial_state,
        )
    )
    return final_subtree_expression, final_index_array, final_index_pointer


@jit
def pick_random_node(
    rng_key: PRNGKeyArray,
    post_level_map: Int[jnp.ndarray, " D"],
) -> ScalarInt:
    """
    Randomly pick a node from the post-order expression, return the
    corresponding node in level order. In the current implementation
    this is done by sampling uniformly from the non-empty nodes.

    The post_level_map contains all indices used in the the level
    order expression, which makes it extremely easy to sample a node
    uniformly at random. It is assumed at empty nodes are represented
    by -1 in the post_level_map, while all other nodes are have
    values >= 0.

    Parameters
    ----------
    rng_key : PRNGKeyArray
        Random key.
    post_level_map :  Int[jnp.ndarray, " D"]
        The map from post-order to level-order.

    Returns
    -------
    ScalarInt
        The randomly chosen node (in level order notation).

    """
    # set weights, mask empty nodes
    weights = jnp.where(post_level_map >= 0, 1.0, 0.0)
    return jr.choice(rng_key, post_level_map, p=weights)


def generate_random_path(
    key: PRNGKeyArray,
    start_idx: ScalarInt,
    node_height: ScalarInt,
    max_depth: ScalarInt,
    stop_probability: ScalarFloat = 0.5,
    go_left_probability: ScalarFloat = 0.5,
) -> tuple[
    ScalarInt,
    Int[jnp.ndarray, " D"],
    ScalarFloat,
]:
    """
    Generate a random path in a binary tree starting from a given index.
    Returns the final chosen index, the path taken and the log probability
    of reaching the chosen index.

    Parameters
    ----------
    key : jr.PRNGKey
        The random key.
    start_idx : ScalarInt
        The index of the starting node.
    max_depth : ScalarInt
        The maximum depth of the tree.
    node_height : ScalarInt
        The height of the subtree at the starting index. This is used
        to calculate the maximum number of levels the path can traverse,
        and still respect the maximum depth of the tree once the kernel
        is reattached.
    stop_probability : ScalarFloat, optional
        The probability of stopping at each level, by default 0.5.
    go_left_probability : ScalarFloat, optional
        The probability of choosing the left child at each
        level, by default 0.5.

    Returns
    -------
    ScalarInt
        The index of the final chosen node (in level order).
    Int[jnp.ndarray, " D"]
        An array of indices representing the path taken, all
        parent indices from the start_idx to the final chosen index
        (in level order), excluding the final index.
    ScalarFloat
        The log probability of reaching the chosen idx. The probability
        is calculated by multiplying the probabilities of stopping at
        at each index along the path and the probabilities
        of choosing the left or right child at each level.
    """

    index_array = jnp.full(max_depth + 1, -1)

    chosen_idx, index_path, num_indices, idx_log_p = _generate_random_path(
        key,
        start_idx,
        max_depth - node_height,
        index_array,
        stop_probability,
        go_left_probability,
    )

    index_path = index_path

    return chosen_idx, index_path, idx_log_p


@jit
def _generate_random_path(
    key: PRNGKeyArray,
    start_idx: ScalarInt,
    max_depth: ScalarInt,
    index_array: Int[jnp.ndarray, " D"],
    stop_probability: ScalarFloat = 0.5,
    go_left_probability: ScalarFloat = 0.5,
) -> tuple[
    ScalarInt,
    Int[jnp.ndarray, " D"],
    ScalarInt,
    ScalarFloat,
]:
    """
    Generate a random path in a binary tree starting from a given index.

    NOTE: This function should be called via the generate_random_path function.

    Parameters
    ----------
    key : jr.PRNGKey
        The random key.
    start_idx : ScalarInt
        The index of the starting node.
    max_depth : ScalarInt
        The maximum depth of the tree.
    index_array :  Int[jnp.ndarray, " D"]
        An empty array to store the path taken, should
        be long enough to store the path to the deepest
        possible node.
    stop_probability : ScalarFloat, optional
        The probability of stopping at each level, by default 0.5.
    go_left_probability : ScalarFloat, optional
        The probability of choosing the left child at each
        level, by default 0.5.

    Returns
    -------
    ScalarInt
        The index of the final chosen node (in level order).
    Int[jnp.ndarray, " D"]
        An array of indices representing the path taken, all
        parent indices from the start_idx to the final chosen index
        (in level order), excluding the final index.
    ScalarInt
        The number of indices in the path (from start_idx to the final
        chosen index).
    ScalarFloat
        The log probability of reaching the chosen idx. The probability
        is calculated by multiplying the probabilities of stopping at
        at each index along the path and the probabilities
        of choosing the left or right child at each level.

    """
    log_stop_probability = jnp.log(stop_probability)
    log_continue_probability = jnp.log(1 - stop_probability)
    log_go_left_probability = jnp.log(go_left_probability)
    log_go_right_probability = jnp.log(1 - go_left_probability)

    def generate_path(state: tuple) -> tuple:
        """Generate a random path in a binary tree."""
        key, idx, index_array, pointer, log_probability, _ = state

        # set the current index in the index array
        index_array = index_array.at[pointer].set(idx)

        key, done_key, direction_key = jr.split(key, 3)

        # evaluate if traversal is done, taking
        # into account if maximum depth is reached
        is_at_max_depth = get_depth(idx) == max_depth
        is_done = jnp.where(
            is_at_max_depth,
            jnp.array(True),
            jr.bernoulli(done_key, p=stop_probability),
        )

        # calculate the probability of stopping at this level,
        # again taking into account if maximum depth is reached
        log_probability = jnp.where(
            is_at_max_depth,
            log_probability,
            jnp.where(
                is_done,
                log_probability + log_stop_probability,
                log_probability + (log_continue_probability),
            ),
        )
        # if the traversal is not done, move to pointer to next node
        pointer = jnp.where(is_done, pointer, pointer + 1)

        # if the traversal is done, return the current index,
        # if not, choose the next index to move to based on the
        # probability of going left or right
        next_idx = jnp.asarray(
            jnp.where(
                is_done,
                idx,
                jnp.where(
                    jr.bernoulli(direction_key, p=go_left_probability),
                    get_child_idx(idx, "l"),
                    get_child_idx(idx, "r"),
                ),
            )
        )
        # check if left or right child was chosen, 0 if right (even idx),
        # 1 if left (odd idx)
        going_left = jnp.mod(next_idx, 2)

        # if traversal is not done, update the probability
        # by multiplying the probability of choosing the left
        # or right child
        log_probability = jnp.where(
            is_done,
            log_probability,
            jnp.where(
                going_left,
                log_probability + log_go_left_probability,
                log_probability + log_go_right_probability,
            ),
        )

        return (key, next_idx, index_array, pointer, log_probability, is_done)

    def condition(state: tuple) -> ScalarBool:
        """Check if generation is done."""
        return ~state[-1]

    initial_state = (key, start_idx, index_array, 0, 0.0, False)

    _, final_idx, index_array, num_indices, final_log_probability, _ = lax.while_loop(
        condition,
        generate_path,
        initial_state,
    )
    return final_idx, index_array, num_indices, final_log_probability


def reconstruct_path(
    start_idx: ScalarInt,
    end_idx: ScalarInt,
    max_depth: ScalarInt,
    stop_probability: ScalarFloat = 0.5,
    go_left_probability: ScalarFloat = 0.5,
) -> tuple[
    Int[jnp.ndarray, " D"],
    ScalarFloat,
]:
    """
    Given the start index and the end index of a path in a binary tree,
    reconstruct the path taken from the start index to the end index and
    calculate the log probability of reaching the end index.
    (This is the inverse of the generate_random_path function.)

    Parameters
    ----------
    start_idx : ScalarInt
        The index of the starting node.
    end_idx : ScalarInt
        The index of the final node.
    max_depth : ScalarInt
        The maximum depth of the tree.
    stop_probability : ScalarFloat, optional
        The probability of stopping at each level, by default 0.5.
    go_left_probability : ScalarFloat, optional
        The probability of choosing the left child at each
        level, by default 0.5.

    Returns
    -------
     Int[jnp.ndarray, " D"]
        An array of indices representing the path taken, all
        parent indices from the start_idx to the final chosen index
        (in level order), excluding the final index.
    ScalarFloat
        The log probability of reaching the chosen idx. The probability
        is calculated by multiplying the probabilities of stopping at
        at each index along the path and the probabilities
        of choosing the left or right child at each level.
    """

    index_array = jnp.full(max_depth + 1, -1)

    path_array, num_indices, log_probability = _reconstruct_path(
        start_idx,
        end_idx,
        max_depth,
        index_array,
        stop_probability,
        go_left_probability,
    )

    # reverse the path array to get the path from start to end
    # (NOTE: To keep array static but only reverse filled part,
    # we reverse the array, then roll last part back to the front).
    path_array = jnp.roll(path_array[::-1], num_indices)
    return path_array, log_probability


@jit
def _reconstruct_path(
    start_idx: ScalarInt,
    end_idx: ScalarInt,
    max_depth: ScalarInt,
    index_array: Int[jnp.ndarray, " D"],
    stop_probability: ScalarFloat = 0.5,
    go_left_probability: ScalarFloat = 0.5,
) -> tuple[
    Int[jnp.ndarray, " D"],
    ScalarInt,
    ScalarFloat,
]:
    """
    Given the start index and the end index of a path in a binary tree,
    reconstruct the path taken from the start index to the end index and
    calculate the log probability of reaching the end index.

    NOTE: This function should be called via the reconstruct_path function.

    Parameters
    ----------
    start_idx : ScalarInt
        The index of the starting node.
    end_idx : ScalarInt
        The index of the final node.
    max_depth : ScalarInt
        The maximum depth of the tree.
    index_array :  Int[jnp.ndarray, " D"]
        An empty array to store the path taken, should
        be long enough to store the path to the deepest
        possible node.
    stop_probability : ScalarFloat, optional
        The probability of stopping at each level, by default 0.5.
    go_left_probability : ScalarFloat, optional
        The probability of choosing the left child at each
        level, by default 0.5.

    Returns
    -------
     Int[jnp.ndarray, " D"]
        An array of indices representing the path taken, all
        parent indices from the start_idx to the final chosen index
        (in level order), excluding the final index.
    ScalarInt
        The number of indices in the path (from start_idx to the final
        chosen index).
    ScalarFloat
        The log probability of reaching the chosen idx. The probability
        is calculated by multiplying the probabilities of stopping at
        at each index along the path and the probabilities
        of choosing the left or right child at each level.

    """
    log_stop_probability = jnp.log(stop_probability)
    log_continue_probability = jnp.log(1 - stop_probability)
    log_go_left_probability = jnp.log(go_left_probability)
    log_go_right_probability = jnp.log(1 - go_left_probability)

    def reconstruct_path(state: tuple) -> tuple:
        """Reconstruct the path from the end index to the start index."""
        current_idx, index_array, pointer, log_probability = state

        # if current index is the end index, don't update the index array
        # and pointer since generate_random_path does not include the end index
        index_array_updated = jnp.where(
            current_idx == end_idx,
            index_array,
            index_array.at[pointer].set(current_idx),
        )
        pointer_updated = jnp.where(
            current_idx == end_idx,
            pointer,
            pointer + 1,
        )

        # check if the current index is at the maximum depth
        # or start, and if its left or right child
        is_at_max_depth = get_depth(current_idx) == max_depth
        is_at_start = current_idx == start_idx
        is_left_node = current_idx % 2  # 0 if right, 1 if left

        # update log probability with the probability of stopping
        # at the current index, taking into account if the current
        # index is at the maximum depth
        log_probability = jnp.where(
            is_at_max_depth,
            log_probability,
            jnp.where(
                current_idx == end_idx,
                log_probability + log_stop_probability,
                log_probability + log_continue_probability,
            ),
        )

        # update log probability with the probability of choosing
        # the left child at the current index, taking into account
        # if the current index is at the start of the path
        log_probability = jnp.where(
            is_at_start,
            log_probability,
            jnp.where(
                is_left_node,
                log_probability + log_go_left_probability,
                log_probability + log_go_right_probability,
            ),
        )

        parent_idx = get_parent_idx(current_idx)

        return (
            parent_idx,
            index_array_updated,
            pointer_updated,
            log_probability,
        )

    def condition(state: tuple) -> ScalarBool:
        """Check if the start index has been reached."""
        return state[0] >= start_idx

    initial_state = (end_idx, index_array, 0, 0.0)

    _, path_array, num_indices, final_log_probability = lax.while_loop(
        condition,
        reconstruct_path,
        initial_state,
    )

    return path_array, num_indices, final_log_probability
