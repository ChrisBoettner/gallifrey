import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import InverseGamma

from gallifrey.data import Dataset
from gallifrey.utils.probability_calculations import calculate_inv_sigma_r
from gallifrey.utils.typing import ScalarFloat


def noise_variance_proposal(
    key: PRNGKeyArray,
    kernel_gram: Float[jnp.ndarray, "D D"],
    noise_variance: ScalarFloat,
    data: Dataset,
) -> tuple[ScalarFloat, ScalarFloat]:
    """
    Sample a new proposal for the noise variance,
    and the log probability of the proposal.

    See equation 21 in Saad et al. 2023.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_gram : Float[jnp.ndarray, "D D"]
        The gram matrix of the kernel.
    noise_variance : ScalarFloat
        The current noise variance.
    data : Dataset
        The data containing the observations,
        as a Dataset object.

    Returns
    -------
    ScalarFloat
        The new noise variance.
    ScalarFloat
        The log probability of the proposal.
    """

    # calculate Σ⁻¹y | [n,1]
    inv_sigma_r = calculate_inv_sigma_r(
        data.y,
        kernel_gram,
        noise_variance,
    )

    # calculate the means of the Gaussian process, µ = Kxx*Σ⁻¹y | [n,1]
    means = kernel_gram @ inv_sigma_r

    # sample new noise variance from inverse gamma distribution
    diff = data.y - means
    inv_gamma = InverseGamma(
        concentration=1 + data.y.size / 2,
        scale=1 + 0.5 * diff @ diff,
    )
    new_noise_variance = inv_gamma.sample(seed=key, sample_shape=())
    log_prob = inv_gamma.log_prob(new_noise_variance)

    return new_noise_variance, log_prob  # type: ignore


def noise_variance_probability(
    kernel_gram: Float[jnp.ndarray, "D D"],
    current_noise_variance: ScalarFloat,
    proposed_noise_variance: ScalarFloat,
    data: Dataset,
) -> ScalarFloat:
    """
    Calculate the posterior probability for a
    proposed noise variance, given the current
    kernel gram matrix and current noise variance.

    See equation 21 in Saad2023.

    Parameters
    ----------
    kernel_gram : Float[jnp.ndarray, "D D"]
        The gram matrix of the kernel.
    current_noise_variance : ScalarFloat
        The current noise variance.
    proposed_noise_variance : ScalarFloat
        The proposed noise variance.
    data : Dataset
        The data containing the observations,
        as a Dataset object.

    Returns
    -------
    ScalarFloat
        The posterior probability for the proposed noise variance.
    """

    # calculate Σ⁻¹y | [n,1]
    inv_sigma_r = calculate_inv_sigma_r(
        data.y,
        kernel_gram,
        current_noise_variance,
    )

    # calculate the means of the Gaussian process, µ = Kxx*Σ⁻¹y | [n,1]
    means = kernel_gram @ inv_sigma_r

    # create inverse gamma distribution
    diff = data.y - means
    inv_gamma = InverseGamma(
        concentration=1 + data.y.size / 2,
        scale=1 + 0.5 * diff @ diff,
    )
    # calculate the log probability of the proposal
    log_prob_proposal = inv_gamma.log_prob(proposed_noise_variance)

    return log_prob_proposal
