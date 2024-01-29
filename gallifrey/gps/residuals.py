import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from gpjax.typing import Array
from jax import jit
from jax.scipy.linalg import cholesky, inv


@jit
def whitened_residuals(
    y: Array,
    distribution: tfp.distributions.Distribution,
) -> Array:
    """Calculate whitened residuals for between
    data y and the predictive distribution through
    Cholesky decomposition.

    Parameters
    ----------
    y : Array
        y data, usually the light curve/flux.
    distribution : tfp.distributions.Distribution
        The predicitve distribution at the points
        x corresponding to y.

    Returns
    -------
    Array
        The whitened residuals.
    """
    residuals = y - distribution.mean()
    covariance_matrix = distribution.covariance()
    cholesky_matrix = cholesky(covariance_matrix)
    inverse_matrix = inv(cholesky_matrix)
    whitened_residuals = jnp.dot(inverse_matrix, residuals)
    return whitened_residuals
