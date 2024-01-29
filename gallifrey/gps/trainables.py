import jax.numpy as jnp
from beartype.typing import Callable
from gpjax.base import meta_map
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_structure
from jaxtyping import Array, install_import_hook
from numpy.typing import NDArray

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


def get_trainables(
    posterior: gpx.gps.AbstractPosterior | gpx.base.Module,
    unconstrain: bool = False,
) -> Array:
    """Print values of trainable parameter
    of model.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior | gpx.base.Module
            The gpjax posterior model.
    unconstrain : bool, optional
        If True, model parameter are pushed trough bijector first
        making their support encompass the entire real line, by
        default False

    Returns
    -------
    Array
        List of trainable parameter values.
    """
    if unconstrain:
        posterior = posterior.unconstrain()

    all_params = ravel_pytree(posterior)[0]
    trainable_mask = ravel_pytree(posterior.trainables())[0]
    return all_params[trainable_mask]


def set_obs_stddev(
    posterior: gpx.gps.AbstractPosterior, obs_stddev: float | Array
) -> gpx.gps.AbstractPosterior:
    """Returns posterior with updated obs_stddev.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
        The gpjax posterior model.
    obs_stddev : float | Array
        The new obs_stddev.

    Returns
    -------
    gpx.gps.AbstractPosterior
        The updated posterior.
    """
    likelihood = posterior.likelihood.replace(obs_stddev=jnp.array(obs_stddev))
    return likelihood * posterior.prior  # type: ignore


def set_trainables(
    posterior: gpx.gps.AbstractPosterior,
    parameter: tuple | list | Array | NDArray,
    unconstrain: bool = False,
) -> gpx.gps.AbstractPosterior:
    """Returns posterior with updated trainable parameter.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
        The gpjax posterior model.
    parameter : tuple | list | Array | NDArray
        The list of new parameter. Must be of same length
        as the number of trainable parameter
    unconstrain : bool, optional
        If True, model parameter are pushed trough bijector first
        making their support encompass the entire real line. The given
        parameter list must then be in this unconstrained space, by
        default False

    Returns
    -------
    gpx.gps.AbstractPosterior
        The posterior with updated parameter.
    """

    def create_parameter_updater() -> Callable:
        # Check if the number of trainable parameters matches the length
        # of the parameter list
        num_trainable_params = sum(ravel_pytree(posterior.trainables())[0])
        if len(parameter) != num_trainable_params:
            raise ValueError(
                f"The length of the parameter list ({len(parameter)}) must "
                "match the number of trainable parameters "
                f"({num_trainable_params}) in the model."
            )

        # create iterable iterates through values in parameter list
        # everytime its called
        param_iterator = iter(parameter)

        # filter leaves, and assign new parameter from parameter list
        # if trainable is found
        def update_trainable(meta_leaf: tuple[dict, Array]) -> Array:
            meta, leaf = meta_leaf
            if meta.get("trainable", False):
                try:
                    return jnp.array(next(param_iterator))
                except StopIteration:
                    raise IndexError(
                        "Found more parameter in paramter list than "
                        "trainable parameters in model."
                    )
            else:
                return leaf

        return update_trainable

    updater = create_parameter_updater()

    if unconstrain:
        posterior = posterior.unconstrain()
    return meta_map(updater, posterior)  # type: ignore


# @jit
def jit_set_trainables(
    posterior: gpx.gps.AbstractPosterior,
    parameter: Array,
    trainable_idx: Array,
) -> gpx.gps.AbstractPosterior:
    """A jit-compatible version of set_trainables. For this, the indices of the
    trainable parameter must be given explicitly. There's no initial check of
    lengths, so make sure the length of the parameter array is the same as the
    number of trainable parameter.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
    parameter : Array
        The array of new parameter. Must be of same length
        as the number of trainable parameter, and a jnp array.
    trainable_idx : Array
        A jnp array containing the indices of the trainable parameter.

    Returns
    -------
    gpx.gps.AbstractPosterior
        The posterior with updated parameter.
    """
    old_parameter = jnp.array(tree_leaves(posterior))
    updated_parameter = old_parameter.at[trainable_idx].set(parameter)

    new_posterior = tree_structure(posterior).unflatten(updated_parameter)
    return new_posterior
