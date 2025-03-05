from functools import partial

import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax import debug, jit, lax
from jaxtyping import Bool, Float, Int, PRNGKeyArray

from gallifrey.kernels.prior import KernelPrior
from gallifrey.moves.subtree_replace import replace_subtree_structure_with
from gallifrey.parameter import move_parameters
from gallifrey.utils.tree_helper import (
    calculate_max_nodes,
    generate_random_path,
    get_child_idx,
    get_depth,
    get_subtree,
    pick_random_node,
    reconstruct_path,
)
from gallifrey.utils.typing import ScalarBool, ScalarFloat, ScalarInt


@partial(jit, static_argnames=("kernel_prior", "verbosity"))
def detach_attach_move(
    key: PRNGKeyArray,
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    detach_prob: ScalarFloat = 0.5,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarFloat]:
    """
    Perform detach-attach move. In this move, a subtree is detached
    from the tree expression (at idx a), and a scaffold is attached
    to the tree expression (at idx a). The subtree is then reattached
    to the scaffold.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_state : nnx.State
        The original kernel state, must have attributes:
        - parameters
        - leaf_level_map
        - node_sizes
        - tree_expression
        - is_operator
    kernel_prior : KernelPrior
        The kernel prior object that contains the kernel structure prior.
    detach_prob : ScalarFloat, optional
        The probability of performing the detach move, by default 0.5.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    nnx.State
        The updated kernel state after the move.
    ScalarFloat
        The log acceptance ratio contribution of the move,
        needs to be added to the log acceptance ratio of the
        subtree-replace move.

    """
    key, selection_subkey, move_subkey = jr.split(key, 3)

    # Cannot perform detach-attach move on kernels with max depth = 1.
    perform_detach = jr.bernoulli(
        selection_subkey,
        jnp.where(
            kernel_state.node_sizes.value[0] == 1,
            jnp.array(0.0),
            jnp.asarray(detach_prob),
        ),
    )
    if verbosity > 2:
        lax.cond(
            perform_detach,
            lambda _: debug.print("Performing detach move."),
            lambda _: debug.print("Performing attach move."),
            perform_detach,
        )
    # ratio of the probabilities of proposing detach and attach moves,
    # needed for acceptance ratio calculation (Saad2023 - Proposition 2)
    log_detach_vs_attach_prob = jnp.log(1 - detach_prob) - jnp.log(detach_prob)

    # perform detach or attach move
    new_kernel_state, log_q_D, log_q_A = lax.cond(
        perform_detach,
        lambda key: detach_move(key, kernel_state, kernel_prior, verbosity=verbosity),
        lambda key: attach_move(key, kernel_state, kernel_prior, verbosity=verbosity),
        move_subkey,
    )

    # calculate the log acceptance ratio contribution for the move
    # (Saad2023 - Proposition 2)
    log_acceptance_ratio_contribution = lax.cond(
        perform_detach,
        lambda log_p: log_p,
        lambda log_p: -log_p,
        log_q_A - log_q_D + log_detach_vs_attach_prob,
    )

    return new_kernel_state, log_acceptance_ratio_contribution


@partial(jit, static_argnames=("kernel_prior", "verbosity"))
def attach_move(
    key: PRNGKeyArray,
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarFloat, ScalarFloat]:
    """
    Perform attach move. In this move, a subtree is detached from the
    tree expression (at idx a), a scaffold is attached to the tree
    expression (at idx a), and the subtree originally at idx a is
    reattached to the scaffold (at idx b).

    Also returns the probabilities associated with the proposals q_A
    (going from the original tree to the new tree using the attach move)
    and q_D (going from the new tree to the original tree using a detach move).

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_state : nnx.State
        The original kernel state, must have attributes:
        - parameters
        - leaf_level_map
        - node_sizes
        - tree_expression
        - is_operator
    kernel_prior : KernelPrior
        The prior to sample the new subtree from.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.


    Returns
    -------
    nnx.State
        The updated kernel state after the move.
    ScalarFloat
        The probability associated with the proposal q_D (b|a, k', theta'),
        i.e. the transition probability from the new tree to the original tree
        using a detach move.
    ScalarFloat
        The probability associated with the proposal q_A (b|a, k, theta),
        i.e. going from the original tree to the new tree using the attach move.

    """
    key, parameter_key, structure_key = jr.split(key, 3)

    # get the new structure and the mapping between the old and new indices
    new_structure, changes, index_mapping, log_p_path, log_p_scaffold, idx_a, _ = (
        structure_attach_move(
            structure_key,
            kernel_state.tree_expression.value,  # type: ignore
            kernel_state.post_level_map.value,  # type: ignore
            kernel_state.node_heights.value,  # type: ignore
            kernel_prior,
            verbosity=verbosity,
        )
    )

    # get attributes of new tree
    new_kernel = nnx.merge(kernel_prior.graphdef, kernel_state)
    new_kernel.init_tree(
        new_structure,
        kernel_prior.kernel_library,
        kernel_prior.max_depth,
        kernel_prior.num_datapoints,
    )

    # move parameters of subtree from old indices to new indices
    _, new_kernel_state = nnx.split(new_kernel)
    new_kernel_state = move_parameters(
        new_kernel_state,
        index_mapping.T,
        kernel_prior.max_depth,
    )

    # sample new kernel parameters, but first remove the nodes
    # from changes that were already moved to a new place (since we
    # want to keep the parameters of the moved nodes and only sample
    # parameters for completely new nodes introduced by scaffold)
    changes = jnp.where(
        jnp.isin(changes, index_mapping[1]),
        -1,
        changes,
    )

    new_kernel_state, log_prob_parameter = kernel_prior.parameter_prior.sample_subset(
        parameter_key,
        new_kernel_state,
        considered_nodes=changes,
    )
    # calculate q_A (b|a, k, theta), i.e. probability associated with the
    # proposal going from the original tree to the new tree using the attach move
    log_q_A = log_p_path + log_p_scaffold + log_prob_parameter

    # calculate q_D (b|a, k', theta'), i.e. the transition probability
    # from the new tree to the original tree using a detach move
    log_q_D = jnp.log(1 / new_kernel.node_sizes.value[idx_a])

    return new_kernel_state, log_q_D, log_q_A


def structure_attach_move(
    key: PRNGKeyArray,
    tree_expression: Int[jnp.ndarray, " D"],
    post_level_map: Int[jnp.ndarray, " D"],
    node_heights: Int[jnp.ndarray, " D"],
    kernel_prior: KernelPrior,
    verbosity: int = 0,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, "2 D"],
    ScalarFloat,
    ScalarFloat,
    ScalarInt,
    ScalarInt,
]:
    """
    Perform attach move on the kernel structure array. In this move,
    a subtree is detached from the tree expression (at idx a), a
    scaffold is attached to the tree expression (at idx a), and the
    subtree originally at idx a is reattached to the scaffold (at idx b).

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, given in level order notation.
    post_level_map : Int[jnp.ndarray, " D"]
        The array that maps the post order index to the level order index.
        (This is useful, since the post_level_map contains all the indices
        of the nodes in the tree, which we can sample).
    node_heights : Int[jnp.ndarray, " D"]
        The heights of each node in the tree. (See
        gallifrey.kernels.tree.TreeKernel for more information.)
    kernel_prior : KernelPrior
        The kernel prior object that contains the kernel structure prior.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The new tree expression with the scaffold attached and the subtree
        reattached.
    Int[jnp.ndarray, " D"]
        The changes in the tree structure, a list of indices that differ
        between the original tree and the new tree.
    Int[jnp.ndarray, "2 D"]
        The mapping between the old and new indices of the nodes in the
        subtree. The first column contains the old indices, and the second
        column contains the new indices.
    ScalarFloat
        The log probability associated with the path to the hole in the
        scaffold.
    ScalarFloat
        The log probability associated with the scaffold.
    ScalarInt
        The index of the root of the subtree that was detached/scaffold
        attached (idx a).
    ScalarInt
        The index where the subtree was reattached (idx b).
    """
    key, idx_a_subkey, path_subkey, scaffold_subkey = jr.split(key, 4)

    # select where the scaffold will be attached
    idx_a = pick_random_node(idx_a_subkey, post_level_map)

    # generate a random path to the hole in the scaffold (this is where
    # the subtree originally at idx_a will be re-attached),
    # the path needs to respect the node height of the detached subtree,
    # since it shouldn't overshoot the max depth when reattached
    idx_b, path, path_log_prob = generate_random_path(
        path_subkey,
        idx_a,
        max_depth=kernel_prior.max_depth,
        node_height=node_heights[idx_a],
    )

    scaffold, scaffold_log_p = scaffold_proposal(
        key,
        max_depth=kernel_prior.kernel_structure_prior.max_depth,
        path_to_hole=path,
        hole_idx=idx_b,
        probs=kernel_prior.kernel_structure_prior.probs,
        is_operator=kernel_prior.kernel_structure_prior.is_operator,
    )

    # transplant the subtree from the original tree at idx_a to the scaffold
    # at idx_b
    scaffold_with_subtree, index_mapping = transplant_subtree(
        tree_expression,
        scaffold,
        idx_a,
        idx_b,
        int(calculate_max_nodes(kernel_prior.max_depth)),
    )

    # replace the subtree at idx_a with the scaffold + subtree at idx_b
    new_tree_expression, changes = replace_subtree_structure_with(
        tree_expression,
        scaffold_with_subtree,
        idx_a,
    )

    if verbosity > 2:
        debug.print("idx_a: {}", idx_a)
        debug.print("idx_b: {}", idx_b)
        debug.print("path: {}", path)
        debug.print("scaffold: {}", scaffold)
        debug.print("scaffold_with_subtree: {}", scaffold_with_subtree)
        debug.print("path_log_prob: {}", path_log_prob)
        debug.print("scaffold_log_p: {}", scaffold_log_p)

    return (
        new_tree_expression,
        changes,
        index_mapping,
        path_log_prob,
        scaffold_log_p,
        idx_a,
        idx_b,
    )


@partial(jit, static_argnames=("kernel_prior", "verbosity"))
def detach_move(
    key: PRNGKeyArray,
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarFloat, ScalarFloat]:
    """
    Perform detach move. In this move, a scaffold is detached
    from the tree expression (at idx a), and a subtree from the
    scaffold (at idx b) is detached and then reattached to the
    root (idx a) of the scaffold.

    Also returns the probabilities associated with the proposals q_D
    (going from the original tree to the new tree using the detach move)
    and q_A (going from the new tree to the original tree using an attach
    move).

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_state : nnx.State
        The original kernel state, must have attributes:
        - parameters
        - leaf_level_map
        - node_sizes
        - tree_expression
        - is_operator
    kernel_prior : KernelPrior
        The prior to sample the new subtree from.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    nnx.State
        The updated kernel state after the move.
    ScalarFloat
        The probability associated with the proposal q_D (b|a, k, theta),
        i.e. going from the original tree to the new tree using the detach move.
    ScalarFloat
        The probability associated with the proposal q_A (b|a, k', theta'),
        i.e. the transition probability from the new tree to the original tree
        using an attach move.

    """
    # get the new structure and the mapping between the old and new indices
    new_structure, index_mapping, idx_a, idx_b = structure_detach_move(
        key,
        kernel_state.tree_expression.value,  # type: ignore
        kernel_state.post_level_map.value,  # type: ignore
        verbosity=verbosity,
    )

    # get attributes of new tree
    new_kernel = kernel_prior.reconstruct_kernel(kernel_state)
    new_kernel.init_tree(
        new_structure,
        kernel_prior.kernel_library,
        kernel_prior.max_depth,
        kernel_prior.num_datapoints,
    )

    # move parameters from old indices to new indices
    _, new_kernel_state = nnx.split(new_kernel)
    new_kernel_state = move_parameters(
        new_kernel_state,
        index_mapping.T,
        kernel_prior.max_depth,
    )

    # calculate q_D (b|a, k, theta), i.e. the probability associated with the
    # detach move going from the original tree to the new tree using the detach move
    # (which is equal to the probability of sampling idx_b given idx_a
    # was sampled, which is simply the inverse of the number of nodes in the
    # subtree, assuming uniform sampling)
    log_q_D = jnp.log(1 / kernel_state.node_sizes.value[idx_a])  # type: ignore

    # calculate q_A (b|a, k', theta'), i.e. involution attach move
    # going from the new tree to the original tree using an attach move
    path, log_p_path = reconstruct_path(
        idx_a,
        idx_b,
        max_depth=kernel_prior.kernel_structure_prior.max_depth,
    )
    # we use the original kernel here, since we are calculating the probability
    # of sampling exactly the (now) missing scaffold, so we have to traverse
    # the scaffold from the original kernel
    log_p_scaffold = kernel_prior.kernel_structure_prior.log_prob_single(
        kernel_state.tree_expression.value,  # type: ignore
        path_to_hole=path,
        hole_idx=idx_b,
    )
    log_q_A = log_p_path + log_p_scaffold

    return new_kernel_state, log_q_D, log_q_A


def structure_detach_move(
    key: PRNGKeyArray,
    tree_expression: Int[jnp.ndarray, " D"],
    post_level_map: Int[jnp.ndarray, " D"],
    empty_node_value: ScalarInt = -1,
    verbosity: int = 0,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, "2 D"],
    ScalarInt,
    ScalarInt,
]:
    """
    Perform detach move on the kernel structure array. In this move,
    a scaffold is detached from the tree expression (at idx a), and a
    subtree from the scaffold (at idx b) is detached and then
    reattached to the root (idx a) of the scaffold.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key for sampling.
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, given in level order notation.
    post_level_map : Int[jnp.ndarray, " D"]
        The array that maps the post order index to the level order index.
        (This is useful, since the post_level_map contains all the indices
        of the nodes in the tree, which we can sample).
    empty_node_value : ScalarInt, optional
        The value to fill the hole with in the subtree, by default -1.
        (Must be negative.)
    verbosity : int, optional
        The verbosity level, by default 0. Debug information is printed
        if verbosity > 2.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The new tree expression with the scaffold removed and the subtree
        reattached.
    Int[jnp.ndarray, "2 D"]
        The mapping between the old and new indices of the nodes in the
        subtree. The first column contains the old indices, and the second
        column contains the new indices.
    ScalarInt
        The index of the root of the scaffold (idx a).
    ScalarInt
        The index of the root of the subtree that was detached from the
        scaffold (idx b).
    """

    key, idx_a_subkey, idx_b_subkey = jr.split(key, 3)

    # choose the root of the scaffold to detach
    idx_a = pick_random_node(idx_a_subkey, post_level_map)

    # detach the subtree
    amputee_tree, subtree_at_a, subtree_at_a_indices = detach_subtree(
        tree_expression,
        idx_a,
        fill_value=empty_node_value,
    )

    # select the root of the subtree to detach from the scaffold
    idx_b = pick_random_node(idx_b_subkey, subtree_at_a_indices)

    # get the subtree from the scaffold and move it to the root of the scaffold,
    # also get mapping between old and new indices
    subtree_at_b, index_mapping = get_subtree(subtree_at_a, idx_b, new_root_idx=idx_a)

    if verbosity > 2:
        debug.print("idx_a: {}", idx_a)
        debug.print("idx_b: {}", idx_b)
        debug.print("Remaining Original Tree: {}", amputee_tree)
        debug.print("Subtree at a: {}", subtree_at_a)
        debug.print("Subtree at b: {}", subtree_at_b)

    # attach the subtree at the root of the scaffold (this assumes the empty nodes in
    # the original tree are -1, and all filled nodes are >= 0)
    new_tree = jnp.where(subtree_at_b > amputee_tree, subtree_at_b, amputee_tree)

    return new_tree, index_mapping, idx_a, idx_b


def scaffold_proposal(
    key: PRNGKeyArray,
    max_depth: ScalarInt,
    path_to_hole: Int[jnp.ndarray, " D"],
    hole_idx: ScalarInt,
    probs: Float[jnp.ndarray, " D"],
    is_operator: Bool[jnp.ndarray, " D"],
    empty_value: ScalarInt = -1,
) -> tuple[
    Int[jnp.ndarray, " D"],
    ScalarFloat,
]:
    """
    Propose the tree structure for the scaffold in the
    attach move. Also returns the log-probability of the proposal.

    The scaffold is a tree structure proposal very similar
    to the samples created from the kernel prior (see
    gallifrey.kernels.prior.TreeStructurePrior). The difference
    is that this prior is conditioned by the path to the hole, meaning
    some branches are fixed to be operator nodes, so that the hole can
    be reached.
    The root of the scaffold is assumed to be first index in the path_to_hole.

    The hole itself is fixed to be a leaf node, and empty (filled by the
    empty_value) in the scaffold.
    NOTE: This means the returned scaffold is not a valid kernel structure
    in itself, the hole must be filled by a valid subtree.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.
    path_to_hole :  Int[jnp.ndarray, " D"]
        The path to the hole in the tree. A list of indices that
        describe the path from the root to the hole (level order
        indices).
    hole_idx : ScalarInt
        The index of the hole in the tree (level order index).
    probs :  Float[jnp.ndarray, " D"]
        The probabilities of sampling each kernel in the library.
    is_operator : Bool[jnp.ndarray, " D"]
        An array that indicates whether each kernel in the library is an operator.
    empty_value : ScalarInt, optional
        The value to fill the hole with in the scaffold, by default -1.

    Returns
    -------
     Int[jnp.ndarray, " D"]
        An array that describes the scaffold structure.
    ScalarFloat
        The log probability associated with the scaffold.
    """
    max_nodes = calculate_max_nodes(max_depth)

    # create sample array to be filled, this will be the output (empty
    # nodes are labeled -1)
    sample = jnp.full(max_nodes, -1)
    # create initial stack: empty except for the root node, start at beginning of path
    initial_stack = jnp.copy(sample).at[0].set(path_to_hole[0])
    pointer = 0  # initial position of the stack pointer
    initial_log_p = 0.0  # initial probability

    initial_state = (key, sample, initial_stack, pointer, initial_log_p)

    return _scaffold_proposal(
        initial_state,
        probs,
        is_operator,
        path_to_hole,
        hole_idx,
        max_depth,
        empty_value,
    )


@jit
def _scaffold_proposal(
    initial_state: tuple,
    probs: Float[jnp.ndarray, " D"],
    is_operator: Bool[jnp.ndarray, " D"],
    path_to_hole: Int[jnp.ndarray, " D"],
    hole_idx: ScalarInt,
    max_depth: ScalarInt,
    empty_value: ScalarInt = -1,
) -> tuple[jnp.ndarray, ScalarFloat]:
    """
    Construct a scaffold proposal for the attach move.

    NOTE: This function is should be called via the scaffold_proposal
    function.

    Parameters
    ----------
    initial_state : tuple
        Initial state for the while loop.
    probs :  Float[jnp.ndarray, " D"]
        The probabilities of sampling each kernel in the library.
    is_operator : Bool[jnp.ndarray, " D"]
        An array that indicates whether each kernel in the library is an operator.
    path_to_hole :  Int[jnp.ndarray, " D"]
        The path to the hole in the tree. A list of indices that
        describe the path from the root to the hole (level order
        indices).
    hole_idx : ScalarInt
        The index of the hole in the tree (level order index).
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.
    empty_value : ScalarInt, optional
        The value to fill the hole with in the scaffold, by
        default -1.


    Returns
    -------
     Int[jnp.ndarray, " D"]
        An array that describes the scaffold structure.
    ScalarFloat
        The log probability associated with the scaffold.
    """

    def traverse_tree(state: tuple) -> tuple:
        """Traverse the tree and sample the scaffold."""
        key, sample, stack, pointer, log_p = state

        # pop the top of the stack
        pos = stack[pointer]
        pointer = pointer - 1

        # compute the depth of the current node
        depth = get_depth(pos)

        # determine if operators are allowed at this depth
        is_at_max_depth = depth >= max_depth
        # mask the probabilities: if at max_depth, disable operator choices
        masked_probs = jnp.where(
            is_at_max_depth,
            jnp.where(is_operator, 0.0, probs),  # if at max depth, mask operators
            probs,  # else use full probs
        )

        # determine if the current node is on path to the hole
        is_on_path = jnp.isin(pos, path_to_hole)
        masked_probs = jnp.where(
            is_on_path,
            jnp.where(
                ~is_operator, 0.0, masked_probs
            ),  # if on path, mask non-operators
            masked_probs,
        )
        masked_probs /= jnp.sum(masked_probs)

        # create subkey for sampling and next iteration
        key, new_key = jr.split(key)

        # sample choice from masked_probs
        choice = jr.categorical(
            new_key,
            logits=jnp.log(masked_probs),
        )

        # check if the current node is the hole
        is_hole = pos == hole_idx

        # if node is the hole, set the choice to the empty value
        choice = jnp.where(
            is_hole,
            empty_value,
            choice,
        )
        sample_updated = sample.at[pos].set(choice)

        # update the probability for the scaffold
        log_p_updated = jnp.where(
            is_hole,
            log_p,
            log_p + jnp.log(masked_probs[choice]),
        )

        # determine if choice is an operator
        is_op = jnp.where(is_hole, False, is_operator[choice])

        def push(stack_and_pointer: tuple) -> tuple:
            """If the node is an operator, push the new
            child nodes onto the stack.
            """
            stack, pointer = stack_and_pointer

            left_child = get_child_idx(pos, "l")
            right_child = get_child_idx(pos, "r")

            stack_updated = (
                stack.at[pointer + 1].set(right_child).at[pointer + 2].set(left_child)
            )
            pointer_updated = pointer + 2
            return stack_updated, pointer_updated

        def dont_push(stack_and_pointer: tuple) -> tuple:
            """If the node is a leaf node, do not modify the stack."""
            return stack_and_pointer

        # Conditionally sample the node or set it as empty based on parent operator
        stack_updated, pointer_updated = lax.cond(
            is_op,
            push,
            dont_push,
            operand=(stack, pointer),
        )

        return (
            new_key,
            sample_updated,
            stack_updated,
            pointer_updated,
            log_p_updated,
        )

    def condition(state: tuple) -> ScalarBool:
        """Loop condition: Continue until stack is empty."""
        _, _, _, pointer, _ = state
        return pointer >= 0

    # execute the while loop
    _, final_sample, _, _, final_log_p = lax.while_loop(
        condition,
        traverse_tree,
        initial_state,
    )
    return final_sample, final_log_p


def detach_subtree(
    tree_expression: Int[jnp.ndarray, " D"],
    subtree_root_idx: ScalarInt,
    fill_value: ScalarInt = -1,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
]:
    """
    Cut a subtree from a tree expression, starting at the subtree root
    node.

    Returns the tree_expression with the subtree removed, the subtree
    expression, and the indices of the nodes in the subtree.

    Parameters
    ----------
    tree_expression : Int[jnp.ndarray, " D"]
        The tree expression, given in level order notation.
    subtree_root_idx : ScalarInt
        The index of node where the subtree is rooted.
    fill_value : ScalarInt, optional
        The value to fill the hole with in the subtree, by default -1.
        (Must be negative.)
    Returns
    -------
    Int[jnp.ndarray, " D"]
        The (level order) tree expression with the subtree removed.
    Int[jnp.ndarray, " D"]
        The (level order) tree expression of the subtree, same length as the
        input tree expression.
    Int[jnp.ndarray, " D"]
        The indices of the nodes in the subtree.

    """
    max_nodes = tree_expression.size

    # if tree_expression[subtree_root_idx] == -1:
    #     raise ValueError("The start index must be a non-empty node.")
    # if fill_value >= 0:
    #     raise ValueError("The fill value must be negative.")

    # create initial state
    subtree_expression = jnp.full(max_nodes, -1)
    index_array = jnp.copy(subtree_expression)
    index_pointer = 0
    stack = jnp.copy(subtree_expression).at[0].set(subtree_root_idx)
    stack_pointer = 0

    initial_state = (
        tree_expression,
        subtree_expression,
        index_array,
        index_pointer,
        stack,
        stack_pointer,
    )

    tree_expression, subtree_expression, index_array, _ = _detach_subtree(
        initial_state,
        max_nodes,
        fill_value,
    )

    return tree_expression, subtree_expression, index_array


@jit
def _detach_subtree(
    initial_state: tuple[
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        Int[jnp.ndarray, " D"],
        ScalarInt,
        Int[jnp.ndarray, " D"],
        ScalarInt,
    ],
    max_nodes: ScalarInt,
    fill_value: ScalarInt = -1,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, " D"],
    ScalarInt,
]:
    """
    Traverse the tree expression in level-order and cut a subtree from
    the tree expression.

    NOTE: This function is not meant to be called directly. Use get_subtree
    instead.

    Parameters
    ----------
    initial_state : tuple
        The initial state of the traversal. Contains the tree expression,
        the subtree expression, the index array, the index pointer, the stack
        and the stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree.
    fill_value : ScalarInt, optional
        The value to fill the hole with in the subtree, by default -1.
        (Must be negative.)

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The (level order) tree expression with the subtree removed.
    Int[jnp.ndarray, " D"]
        The (level order) tree expression of the subtree, same length as the
        input tree expression.
    Int[jnp.ndarray, " D"]
        The indices of the nodes in the subtree.
    ScalarInt
        The number of nodes in the subtree.

    """

    def traverse(state: tuple) -> tuple:
        """Traverse the tree expression in level-order."""
        (
            tree_expression,
            subtree_expression,
            index_array,
            index_pointer,
            old_stack,
            stack_pointer,
        ) = state

        # get old and new indices
        idx = old_stack[stack_pointer]

        # fill index array with the (new) index of the current node
        index_array_updated = index_array.at[index_pointer].set(idx)
        index_pointer_updated = index_pointer + 1

        # fill the new tree expression with the current node (at the new index)
        subtree_expression_updated = subtree_expression.at[idx].set(
            tree_expression[idx]
        )
        # remove node from original tree
        tree_expression_updated = tree_expression.at[idx].set(fill_value)

        # check if the current node is a leaf
        right_child_idx = get_child_idx(idx, "r")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            tree_expression[right_child_idx - 1] == -1,
        )

        def process_leaf(stacks_and_pointer: tuple) -> tuple:
            """If the current node is a leaf, move to the parent node."""
            old_stack, pointer = stacks_and_pointer
            return old_stack, pointer - 1

        def process_non_leaf(stacks_and_pointer: tuple) -> tuple:
            """If the current node is not a leaf, update the stacks
            (add the children of the current node)."""
            old_stack, pointer = stacks_and_pointer

            left_child_idx = get_child_idx(idx, "l")
            right_child_idx = get_child_idx(idx, "r")

            old_stack_updated = old_stack.at[pointer].set(left_child_idx)
            old_stack_updated = old_stack_updated.at[pointer + 1].set(right_child_idx)

            return old_stack_updated, pointer + 1

        # update the stack and pointer
        old_stack_updated, pointer_updated = lax.cond(
            is_leaf,
            process_leaf,
            process_non_leaf,
            operand=(old_stack, stack_pointer),
        )

        return (
            tree_expression_updated,
            subtree_expression_updated,
            index_array_updated,
            index_pointer_updated,
            old_stack_updated,
            pointer_updated,
        )

    def condition(state: tuple) -> ScalarBool:
        """Check if the traversal is complete."""
        _, _, _, _, _, stack_pointer = state
        return stack_pointer >= 0

    # iterate over the sample using lax.while_loop
    (
        final_tree_expression,
        final_subtree_expression,
        final_index_array,
        final_index_pointer,
        _,
        _,
    ) = lax.while_loop(
        condition,
        traverse,
        initial_state,
    )
    return (
        final_tree_expression,
        final_subtree_expression,
        final_index_array,
        final_index_pointer,
    )


@partial(jit, static_argnames=("max_nodes",))
def transplant_subtree(
    donor_tree_expression: Int[jnp.ndarray, " D"],
    recipient_tree_expression: Int[jnp.ndarray, " D"],
    donor_root_idx: ScalarInt,
    recipient_root_idx: ScalarInt,
    max_nodes: int,
) -> tuple[
    Int[jnp.ndarray, " D"],
    Int[jnp.ndarray, "2 D"],
]:
    """
    Transplant a subtree from one tree to another. The subtree is
    rooted at the donor root index and re-rooted at the recipient
    root index.
    The recipient tree must have an empty node at the recipient root index.

    Parameters
    ----------
    donor_tree_expression : Int[jnp.ndarray, " D"]
        The tree expression of the donor tree.
    recipient_tree_expression : Int[jnp.ndarray, " D"]
        The tree expression of the recipient tree.
    donor_root_idx : ScalarInt
        The index of the root of the subtree to be moved.
    recipient_root_idx : ScalarInt
        The index of the root of the subtree to be moved to.
    max_nodes : int
        The maximum number of nodes in the tree.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The tree expression of the recipient tree with the attached subtree.
    Int[jnp.ndarray, "2 D"]
        The indices of the moved nodes. Two rows, the first row
        contains the indices of the subtree in the donor tree, the second
        row contains the indices of the subtree in the recipient tree.

    """
    # if donor_tree_expression[donor_root_idx] == -1:
    #     raise ValueError("The donor root index must be a non-empty node.")
    # if recipient_tree_expression[recipient_root_idx] != -1:
    #     raise ValueError("The recipient root index must be an empty node.")

    # create initial state
    index_array = jnp.full((2, max_nodes), -1)
    index_pointer = 0
    stack = jnp.full(max_nodes, -1).at[0].set(donor_root_idx)
    new_stack = jnp.copy(stack).at[0].set(recipient_root_idx)
    stack_pointer = 0

    initial_state = (
        recipient_tree_expression,
        index_array,
        index_pointer,
        stack,
        new_stack,
        stack_pointer,
    )

    tree_expression, index_array, _ = _transplant_subtree(
        donor_tree_expression,
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
    return tree_expression, index_array


@jit
def _transplant_subtree(
    donor_tree_expression: Int[jnp.ndarray, " D"],
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
    Traverse the tree expression in level-order, transplanting the subtree
    rooted at the donor root index to the recipient root index.

    NOTE: This function is not meant to be called directly. Use transplant_subtree
    instead.

    Parameters
    ----------
    donor_tree_expression : Int[jnp.ndarray, " D"]
        The tree expression of the donor tree.
    initial_state : tuple
        The initial state of the traversal. Contains the tree expression of the
        recipient tree, the index array, the index pointer, the stack of the
        nodes to be visited in the donor tree, the stack of the nodes to be visited
        in the recipient tree, and the stack pointer.
    max_nodes : ScalarInt
        The maximum number of nodes in the tree.

    Returns
    -------
    Int[jnp.ndarray, " D"]
        The (level order) tree expression of the recipient tree with
        the attached subtree.
    Int[jnp.ndarray, " D"]
        The indices of the moved nodes. Two rows, the first row
        contains the indices of the subtree in the donor tree, the second
        row contains the indices of the subtree in the recipient tree.
    ScalarInt
        The number of indices in the index array.

    """

    def traverse(state: tuple) -> tuple:
        """Traverse the tree expression in level-order."""
        (
            recipient_tree_expression,
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

        # move the current node to the new position
        recipient_tree_expression_updated = recipient_tree_expression.at[new_idx].set(
            donor_tree_expression[old_idx]
        )

        # check if the current node is a leaf
        right_child_idx = get_child_idx(old_idx, "r")
        is_leaf = jnp.logical_or(
            right_child_idx > max_nodes - 1,
            donor_tree_expression[right_child_idx - 1] == -1,
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
            recipient_tree_expression_updated,
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
    final_tree_expression, final_index_array, final_index_pointer, _, _, _ = (
        lax.while_loop(
            condition,
            traverse,
            initial_state,
        )
    )
    return final_tree_expression, final_index_array, final_index_pointer
