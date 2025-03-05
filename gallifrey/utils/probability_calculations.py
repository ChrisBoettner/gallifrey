from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.scipy import linalg
from jaxtyping import Float

from gallifrey.data import Dataset
from gallifrey.utils.typing import ScalarFloat


# @partial(jit, static_argnames=("jitter", "data"))
def calculate_marginal_log_likelihood(
    kernel_gram: Float[jnp.ndarray, "D D"],
    noise_variance: ScalarFloat,
    data: Dataset,
    jitter: float = 1e-6,
) -> ScalarFloat:
    """
    Calculate the log marginal likelihood p(y) for the
    state (k, theta, y, noise).

    Parameters
    ----------
    kernel_gram :  Float[jnp.ndarray, "D D"]
        The kernel gram matrix.
    noise_variance : ScalarFloat
        The noise variance.
    data : Dataset
        A dataset object containing the input x and output y.
    jitter : float, optional
        The jitter term to add to the kernel gram matrix, by default 1e-6.
        (This is to ensure the matrix is positive definite.)

    Returns
    -------
    ScalarFloat
        The log marginal likelihood p(y) for the current state.

    """

    sigma_logdet, sigma_inv_r = calculate_sigma_logdet_and_inv_sigma_r(
        data.y,
        kernel_gram,
        noise_variance,
        jitter,
    )
    # compute marginal log likelihood, -1/2[ n log(2π) + log|Σ| + ( rᵀ Σ⁻¹ r ]
    log_prob = -0.5 * (
        data.n * jnp.log(2.0 * jnp.pi) + sigma_logdet + data.y.T @ sigma_inv_r
    )
    return log_prob


@partial(jit, static_argnames=("jitter"))
def calculate_sigma_logdet_and_inv_sigma_r(
    r: Float[jnp.ndarray, " D"],
    kernel_gram: Float[jnp.ndarray, " D D"],
    noise_variance: ScalarFloat,
    jitter: float = 1e-6,
) -> tuple[ScalarFloat, Float[jnp.ndarray, " D"]]:
    """
    Calculate the log determinant of the covariance matrix Σ for the
    Gaussian process, and solve for Σ⁻¹r.

    Parameters
    ----------
    r :  Float[jnp.ndarray, " D"]
        The input observational residuals
        (observations y - deterministic mean function).
    kernel_gram : Float[jnp.ndarray, "D D"]
        The kernel gram matrix.
    noise_variance : ScalarFloat
        The noise variance.
    jitter : float, optional
        The jitter term to add to the kernel gram matrix, by default 1e-6.
        (This is to ensure the matrix is positive definite.)

    Returns
    -------
    ScalarFloat
        The log determinant of the covariance matrix Σ.
    Float[jnp.ndarray, " D"]
        The solution for Σ⁻¹r.
    """

    # calculate Sigma matrix for GP, Σ = (Kxx + Io²) | [n,n]
    sigma_matrix = calculate_covariance_matrix(
        kernel_gram,
        noise_variance,
        jitter,
    )

    # solve for Σ⁻¹r | [n,1] using cholesky decomposition,
    # where Σ⁻¹r = L⁻ᵀ L⁻¹ r, with L = cholesky(Σ)
    cho_fac = linalg.cho_factor(sigma_matrix)
    sigma_inv_r = linalg.cho_solve(cho_fac, r)

    # calculate log determinant | [1,1]
    # log(|Σ|) =  log(|L|**2) = 2 log(|L|), with L = cholesky(Σ)
    sigma_logdet = jnp.sum(jnp.log(jnp.diag(cho_fac[0]))) * 2.0
    return sigma_logdet, sigma_inv_r


@partial(jit, static_argnames=("jitter"))
def calculate_inv_sigma_r(
    r: Float[jnp.ndarray, " D"],
    kernel_gram: Float[jnp.ndarray, " D D"],
    noise_variance: ScalarFloat,
    jitter: float = 1e-6,
) -> Float[jnp.ndarray, " D"]:
    """
    Calculate the covariance matrix Σ for the
    Gaussian process, and solve for Σ⁻¹r.

    Parameters
    ----------
    r :  Float[jnp.ndarray, " D"]
        The input observational residuals
        (observations y - deterministic mean function).
    kernel_gram : Float[jnp.ndarray, "D D"]
        The kernel gram matrix.
    noise_variance : ScalarFloat
        The noise variance.
    jitter : float, optional
        The jitter term to add to the kernel gram matrix, by default 1e-6.
        (This is to ensure the matrix is positive definite.)

    Returns
    -------
     Float[jnp.ndarray, " D"]
        The solution for Σ⁻¹r.
    """

    # calculate Sigma matrix for GP, Σ = (Kxx + Io²) | [n,n]
    sigma_matrix = calculate_covariance_matrix(
        kernel_gram,
        noise_variance,
        jitter,
    )
    # solve for Σ⁻¹r | [n,1] using cholesky decomposition,
    # where Σ⁻¹r = L⁻ᵀ L⁻¹ r, with L = cholesky(Σ)
    cho_fac = linalg.cho_factor(sigma_matrix)
    sigma_inv_r = linalg.cho_solve(cho_fac, r)
    return sigma_inv_r


@partial(jit, static_argnames=("jitter"))
def calculate_covariance_matrix(
    kernel_gram: Float[jnp.ndarray, " D D"],
    noise_variance: ScalarFloat,
    jitter: float = 1e-6,
) -> Float[jnp.ndarray, "D D"]:
    """
    Calculate the covariance matrix for a Gaussian process.

    Parameters
    ----------
    kernel_gram : LinearOperator
        The kernel gram matrix.
    noise_variance : ScalarFloat
        The noise variance.
    jitter : float, optional
        The jitter term to add to the kernel gram matrix, by default 1e-6.
        (This is to ensure the matrix is positive definite.)

    Returns
    -------
    Float[jnp.ndarray, "D D"]
        The covariance matrix for the Gaussian process.

    """
    return kernel_gram + jnp.eye(kernel_gram.shape[0]) * (noise_variance + jitter)
