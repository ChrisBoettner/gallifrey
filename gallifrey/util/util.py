import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def dict_to_jnp(input: ArrayLike | dict) -> Array:
    """Converts dictonary keys to jnp array, also
    converts all other ArrayLike's is possible.

    Parameters
    ----------
    input : ArrayLike | dict
        Input dict or ArrayLike to convert.

    Returns
    -------
    Array
        Output Array. If input was dict, contains
        the values.
    """
    try:
        input = jnp.asarray(input)
    except TypeError:
        if isinstance(input, dict):
            input = jnp.array(list(input.values()))
        else:
            raise ValueError("'input' must be dict or ArrayLike.")
    return input
