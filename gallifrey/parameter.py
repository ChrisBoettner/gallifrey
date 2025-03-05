from functools import partial

import beartype.typing as tp
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from equinox.internal import while_loop as bounded_while_loop
from flax import nnx
from jax import jit, lax
from jaxtyping import Bool, Float, Int

from gallifrey.utils.tree_helper import calculate_max_nodes, get_parameter_leaf_idx
from gallifrey.utils.typing import ScalarBool, ScalarFloat, ScalarInt


class ParticleParameter(nnx.Variable):
    """
    A class to define the particle parameter, which
    is a subclass of the `nnx.Variable` class.
    """

    pass


class KernelParameter(ParticleParameter):
    """
    A class to define the kernel parameter, which
    is a subclass of the `nnx.Variable` class.
    Similar to the Parameter class in GPJax, but holds
    an array of kernel parameters.

    """

    def __init__(
        self,
        value: Float[jnp.ndarray, "M N"],
        **kwargs: tp.Any,
    ):
        """
        Initialize the ParticleParameter class.

        Parameters
        ----------
        value : Float[jnp.ndarray, "M N"]
            The array of kernel parameters, of shape
            (max_leaves, max_atom_parameter)

        """
        super().__init__(
            value=jnp.asarray(value),
            **kwargs,
        )


class NoiseParameter(ParticleParameter):
    """
    Class to define the noise variance parameter.

    """

    def __init__(
        self,
        noise_variance: ScalarFloat | ScalarInt,
        **kwargs: tp.Any,
    ):
        """
        Initialize the ParticleParameter class.

        Parameters
        ----------
        noise_variance : ScalarFloat | ScalarInt
            The value of the noise variance.

        """
        super().__init__(
            value=jnp.asarray(noise_variance),
            **kwargs,
        )


@partial(jit, static_argnames=("max_depth", "reset_unused"))
def move_parameters(
    kernel_state: nnx.State,
    mapping: Int[jnp.ndarray, "M N"],
    max_depth: int,
    reset_unused: bool = True,
) -> nnx.State:
    """
    A function to move parameters from one leaf
    to another based on the mapping.

    Parameters
    ----------
    kernel_state : nnx.State
        The original kernel state, must contain the
        attribute:
        - parameters
        - max_depth
        - tree_expression
        - is_operator
    mapping : Float[jnp.ndarray, "M N"]
        The mapping between the old and new (level order) indices of the nodes
        in the tree. The first column contains the old indices, and the second
        column contains the new indices.
    max_depth : int
        The maximum depth of the tree.
    reset_unused : bool, optional
        Whether to reset the unused parameters to a default fill value,
        by default True.

    Returns
    -------
    nnx.State
        The kernel state with the moved parameters.
    """
    # get needed state attributes
    max_nodes = int(calculate_max_nodes(max_depth))
    tree_expression: jnp.ndarray = kernel_state.tree_expression.value  # type: ignore
    is_operator: jnp.ndarray = kernel_state.is_operator.value  # type: ignore

    # get two versions of the kernel parameters, one to update and
    # to use for copying (in case some leaves are overwritten, before
    # they are moved).
    old_kernel_params = kernel_state["parameters"].value
    new_kernel_params = old_kernel_params.copy()  # type: ignore

    def move_leaf_params(
        new_kernel_params: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        """Inner function to move the leaf parameters."""

        old_idx, new_idx = indices

        # check whether the moved node is an operator or an atom,
        # only atoms have parameters that can be moved
        is_atom = is_operator[tree_expression[new_idx]]

        def do_move(new_kernel_params: jnp.ndarray) -> jnp.ndarray:
            """Move parameters for atom nodes."""

            # get old and new leaf indices
            new_leaf_idx = get_parameter_leaf_idx(new_idx, max_depth)
            old_leaf_idx = get_parameter_leaf_idx(old_idx, max_depth)

            # map old parameter to new location
            new_kernel_params = new_kernel_params.at[new_leaf_idx].set(
                old_kernel_params[old_leaf_idx]
            )

            return new_kernel_params

        def dont_move(new_kernel_params: jnp.ndarray) -> jnp.ndarray:
            """Do not move parameters for operator nodes."""
            return new_kernel_params

        new_kernel_params = lax.cond(
            is_atom,
            lambda x: dont_move(x),
            lambda x: do_move(x),
            new_kernel_params,
        )
        return new_kernel_params

    def loop_body(loop_state: tuple) -> tuple:
        """Loop over indices in mapping, update kernel parameters."""
        new_kernel_params, i = loop_state
        indices = mapping[i]
        new_kernel_params = move_leaf_params(new_kernel_params, indices)
        return new_kernel_params, i + 1

    def condition(loop_state: tuple) -> Bool[jnp.ndarray, " "]:
        """Break when all moves are done, assuming empty slots are
        indicated by negative indices (and that mapping is sorted)."""
        _, i = loop_state
        indices = mapping[i]
        return indices[0] >= 0

    new_kernel_params, _ = bounded_while_loop(
        condition,
        loop_body,
        (new_kernel_params, 0),
        max_steps=max_nodes,
        kind="bounded",
    )

    # update kernel state
    kernel_state["parameters"] = kernel_state["parameters"].replace(  # type: ignore
        new_kernel_params,
    )

    if reset_unused:
        # reset now empty leaf parameter back to -1
        kernel_state = reset_unused_parameters(
            kernel_state,
            max_nodes,
        )

    return kernel_state


@partial(jit, static_argnames=("max_leaves"))
def reset_unused_parameters(
    kernel_state: nnx.State,
    max_leaves: int,
    fill_value: ScalarFloat = -1.0,
) -> nnx.State:
    """
    Reset parameter in parmaeter array for unused leaf nodes
    with default fill value. Leaves might still contain values
    after e.g. a move operation, this function resets them.

    Parameters
    ----------
    kernel_state : nnx.State
        The kernel state, must contain the attribute:
        - parameters
        - leaf_level_map
    max_leaves : int
        The maximum number of leaf nodes.
    fill_value : ScalarFloat, optional
        The new value to fill the leaves with, by default -1.0.

    Returns
    -------
    nnx.State
        The kernel state with the reset parameters.
    """

    leaf_level_map: jnp.ndarray = kernel_state.leaf_level_map.value  # type: ignore
    kernel_params: jnp.ndarray = kernel_state.parameters.value  # type: ignore

    def reset_scan_func(kernel_params: jnp.ndarray, i: ScalarInt) -> tuple:
        """Scan over leafs, if unused reset to fill_value."""
        kernel_params = lax.cond(
            leaf_level_map[i] >= 0,
            lambda: kernel_params,
            lambda: kernel_params.at[i].set(fill_value),
        )
        return kernel_params, i

    kernel_params, _ = lax.scan(
        reset_scan_func,
        kernel_params,
        jnp.arange(max_leaves),
    )

    kernel_state["parameters"] = kernel_state["parameters"].replace(  # type: ignore
        kernel_params,
    )
    return kernel_state


@partial(
    jit,
    static_argnames=(
        "max_leaves",
        "max_atom_parameters",
        "bijectors",
    ),
)
def transform_kernel_parameters(
    kernel_state: nnx.State,
    num_parameter_array: Int[jnp.ndarray, " D"],
    max_leaves: int,
    max_atom_parameters: int,
    support_mapping_array: Int[jnp.ndarray, "M N"],
    bijectors: tuple[tfb.Bijector, ...],
    inverse: ScalarBool = False,
) -> nnx.State:
    """
    A function takes a kernel state and transforms its parameter
    based on the support mapping and bijectors.

    Parameters
    ----------
    kernel_state : nnx.State
        The original kernel state to be filled with new parameters.
        The state must contain the following attributes:
        - tree_expression: The tree expression that describes the kernel structure.
        - leaf_level_map: A array that maps the index of the leaf parameter array
        - parameters: The kernel parameters to be transformed.
    num_parameter_array : Int[jnp.ndarray, " D"]
        An array that contains the number of parameters for each
        atom in the kernel structure. Needs to be in same order
        as the atom library.
    max_leaves : int
        The maximum number of leaves in the tree.
    max_atom_parameters : int
        The maximum number of parameters any atom will take.
    support_mapping_array : Int[jnp.ndarray, "M N"]
        An array that maps the support of the parameters to the
        corresponding bijector index.
    bijectors : tuple[tp.Bijector, ...]
        A tuple of bijectors to transform the parameters to the
        desired support. Must be tensorflow bijectors with forward
        and inverse methods.
    inverse : ScalarBool, optional
        If True, the inverse transformation is applied, by default False.

    Returns
    -------
    nnx.State
        The state with transformed parameters.

    """

    tree_expression: jnp.ndarray = kernel_state.tree_expression.value  # type: ignore
    leaf_level_map: jnp.ndarray = kernel_state.leaf_level_map.value  # type: ignore

    forward_bijectors = [bijector.forward for bijector in bijectors]
    inverse_bijectors = [bijector.inverse for bijector in bijectors]

    def transform_forward(idx_and_param: tuple[ScalarInt, ScalarFloat]) -> ScalarFloat:
        """Forward transform based on idx (in support array)."""
        idx, param = idx_and_param
        return lax.switch(
            idx,
            forward_bijectors,
            param,
        )

    def transform_inverse(idx_and_param: tuple[ScalarInt, ScalarFloat]) -> ScalarFloat:
        """Inverse transform based on idx (in support array)."""
        idx, param = idx_and_param
        return lax.switch(
            idx,
            inverse_bijectors,
            param,
        )

    def process_params(
        kernel_parameters: Float[jnp.ndarray, "M N"],
        indices: Int[jnp.ndarray, "2"],
    ) -> tuple[
        Float[jnp.ndarray, "M N"],
        Int[jnp.ndarray, "2"],
    ]:
        """Process parameter based on if it's active and considered."""

        # unpack indices
        leaf_idx, parameter_idx = indices

        # check whether the parameter is active
        node_value = tree_expression[leaf_level_map[leaf_idx]]
        atom_num_parameters = num_parameter_array[node_value]

        is_active_node = leaf_level_map[leaf_idx] >= 0
        is_active_param = parameter_idx < atom_num_parameters
        is_active = is_active_node & is_active_param

        def process_active(indices: tuple) -> ScalarFloat:
            """If active, transform parameter."""
            leaf_idx, parameter_idx = indices

            # select parameter and transform back to standard normal
            param = kernel_parameters[leaf_idx, parameter_idx]

            transformed_param = lax.cond(
                inverse,
                transform_inverse,
                transform_forward,
                (support_mapping_array[node_value, parameter_idx], param),
            )
            return transformed_param

        def process_inactive(indices: tuple) -> ScalarFloat:
            """If inactive, return as is."""
            return kernel_parameters[leaf_idx, parameter_idx]

        # process parameter, get transformed parameter
        transformed_parameter = lax.cond(
            is_active,
            process_active,
            process_inactive,
            (leaf_idx, parameter_idx),
        )

        # update kernel parameter
        kernel_parameters = kernel_parameters.at[leaf_idx, parameter_idx].set(
            transformed_parameter
        )

        return kernel_parameters, indices

    # get initial kernel parameters
    kernel_parameters: jnp.ndarray = kernel_state.parameters.value  # type: ignore

    # get all parameter index combinations
    parameter_indices = jnp.indices((max_leaves, max_atom_parameters)).reshape(2, -1).T

    # scan over all parameters and update kernel parameters
    transformed_kernel_parameters, _ = lax.scan(
        process_params,
        kernel_parameters,
        parameter_indices,
    )

    kernel_state["parameters"] = kernel_state["parameters"].replace(
        transformed_kernel_parameters,
    )  # type: ignore

    return kernel_state
