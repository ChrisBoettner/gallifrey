from functools import partial

import jax.numpy as jnp
from flax import nnx
from jax import debug, jit, lax
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.kernels.prior import KernelPrior
from gallifrey.moves.noise_proposal import noise_variance_probability
from gallifrey.utils.probability_calculations import calculate_marginal_log_likelihood
from gallifrey.utils.typing import ScalarFloat


@jit
def calculate_acceptance_probability(
    hastings_ratio_numerator: ScalarFloat,
    hastings_ratio_denominator: ScalarFloat,
) -> ScalarFloat:
    """
    Calculate the acceptance ratio for a given pair of
    hastings ratio terms.
    In the context of the particle move, the numerator
    is associated with the proposed state, and the denominator
    with the current state.

    Parameters
    ----------
    hastings_ratio_numerator : ScalarFloat
        The hastings ratio contribution for the proposed state.
    hastings_ratio_denominator : ScalarFloat
        The hastings ratio contribution for the current state.
    Returns
    -------
    ScalarFloat
        The MCMC acceptance probability.
    """

    # dealing with NaNs in such a way that the kernel with NaN is rejected,
    # if both are NaN (shouldn't happen), accept proposal and hope for the best
    # NOTE: in general, NaNs should not occur, but they can occur in the
    # kernel gram matrix if the kernel is not positive definite (e.g. due to
    # numerical instabilities)

    log_hastings_ratio = jnp.nan_to_num(
        hastings_ratio_numerator, nan=-jnp.inf
    ) - jnp.nan_to_num(hastings_ratio_denominator, nan=-jnp.inf)
    log_hastings_ratio = jnp.nan_to_num(log_hastings_ratio, nan=0.0)

    acceptance_probability = jnp.clip(
        jnp.exp(log_hastings_ratio),
        max=1.0,
    )

    return acceptance_probability


@partial(
    jit,
    static_argnames=(
        "kernel_prior",
        "data",
        "fix_noise",
        "verbosity",
    ),
)
def calculate_acceptance_prob_contribution(
    particle_state: nnx.State,
    proposed_noise_variance: ScalarFloat,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    data: Dataset,
    fix_noise: bool,
    verbosity: int = 0,
) -> ScalarFloat:
    """
    Calculate the contribution to the acceptance probability
    for a specific particle state and noise variance proposal.

    Parameters
    ----------
    particle_state : nnx.State
        The state of the particle.
    proposed_noise_variance : ScalarFloat
        The noise variance of the proposed state. (which is not
        the current state!)
    kernel_prior : KernelPrior
        The kernel prior.
    noise_prior : Distribution
        The prior distribution of the noise variance.
    data : Dataset
        The observational data.
    fix_noise : bool
        Whether to fix the noise variance is fixed or not.
    verbosity : int, optional
        The verbosity level, by default 0. Debugging information
        is printed if `verbosity > 1`.

    Returns
    -------
    ScalarFloat
        The contribution to the acceptance probability for the
        given particle state and noise variance proposal.

    """

    # unpack particle state
    kernel_state = particle_state.kernel
    current_noise_variance = jnp.asarray(particle_state.noise_variance.value)

    kernel = kernel_prior.reconstruct_kernel(kernel_state)

    # calculate gram matrix for kernel
    kernel_gram = kernel._gram_train(data.x)

    # calculate acceptance probability terms
    log_tree_size = jnp.log(kernel.node_sizes[0])
    marginal_log_likelihood = calculate_marginal_log_likelihood(
        kernel_gram,
        current_noise_variance,
        data,
    )
    log_prob_noise_prior = lax.cond(
        fix_noise,
        lambda x: jnp.array(0.0),
        lambda x: jnp.asarray(
            noise_prior.log_prob(x),
            dtype=data.x.dtype,
        ).squeeze(),
        current_noise_variance,
    )

    # calculate noise posterior probability
    # NOTE: We have to calculate the probability of the current
    # noise variance coming from the proposed noise variance,
    # so the two are swapped in the function call.
    # (eq 21 and 22 in Saad2023)
    log_prob_noise = lax.cond(
        fix_noise,
        lambda: jnp.array(0.0),
        lambda: noise_variance_probability(
            kernel_gram=kernel_gram,
            current_noise_variance=proposed_noise_variance,
            proposed_noise_variance=current_noise_variance,
            data=data,
        ),
    )
    # calculate acceptance probability contribution,
    # (eq 22. and Preposition 2 in Saad2023)
    log_acceptance_prob_contribution = jnp.array(
        [
            marginal_log_likelihood,
            -log_tree_size,
            log_prob_noise_prior,
            -log_prob_noise,
        ]
    ).sum()

    if verbosity > 1:
        debug.print("Current Terms:")
        debug.print("Marginal log likelihood: {}", marginal_log_likelihood)
        debug.print("Log (tree size^-1): {}", -log_tree_size)
        debug.print("Log noise prior probability: {}", log_prob_noise_prior)
        debug.print("Log (probability of noise posterior^-1): {}", -log_prob_noise)
        debug.print("Sum of terms: {}", log_acceptance_prob_contribution)
        debug.print("-" * 50)

    return log_acceptance_prob_contribution
