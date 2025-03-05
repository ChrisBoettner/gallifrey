from copy import deepcopy
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax import debug, jit, lax
from jaxtyping import PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.kernels.prior import KernelPrior
from gallifrey.moves.acceptance_probability import (
    calculate_acceptance_prob_contribution,
    calculate_acceptance_probability,
)
from gallifrey.moves.detach_attach import detach_attach_move
from gallifrey.moves.noise_proposal import noise_variance_proposal
from gallifrey.moves.subtree_replace import subtree_replace_move
from gallifrey.utils.probability_calculations import calculate_marginal_log_likelihood
from gallifrey.utils.typing import ScalarBool, ScalarFloat


@partial(
    jit,
    static_argnames=(
        "kernel_prior",
        "data",
        "fix_noise",
        "verbosity",
    ),
)
def structure_move(
    key: PRNGKeyArray,
    particle_state: nnx.State,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    data: Dataset,
    fix_noise: bool,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarBool]:
    """
    Perform a structure move on the particle state,
    by proposing a new kernel structure (via a detach-attach
    or subtree-replace move), and a new noise variance.

    The acceptance probability is calculated based on the current
    and proposed particle states, and the proposal is accepted
    stochasticly based on that.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key for the move.
    particle_state : nnx.State
        The current particle state.
    kernel_prior : KernelPrior
        The kernel prior used for sampling a new
        kernel structure.
    noise_prior : Distribution
        The noise variance prior, used for calculating
        the contribution of the noise variance to the
        acceptance probability.
    data : Dataset
        The data, a Dataset object containing the
        observations. (The observations y should already
        be centered around the deterministic mean function,
        i.e. it should be the residuals.)
    fix_noise : bool
        Whether to fix the noise variance, or sample it.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information
        is printed if verbosity > 1.

    Returns
    -------
    nnx.State
        The new particle state after the move.
    ScalarBool
        Whether the move was accepted.
    """

    key, proposal_key, acceptance_key = jr.split(key, 3)

    proposed_particle_state, log_hastings_ratio_numerator = particle_state_proposal(
        proposal_key,
        particle_state,
        kernel_prior,
        noise_prior,
        data,
        fix_noise=fix_noise,
        verbosity=verbosity,
    )

    log_hastings_ratio_denominator = calculate_acceptance_prob_contribution(
        particle_state,
        proposed_particle_state.noise_variance.value,  # type: ignore
        kernel_prior,
        noise_prior,
        data,
        fix_noise=fix_noise,
        verbosity=verbosity,
    )

    acceptance_probability = calculate_acceptance_probability(
        log_hastings_ratio_numerator,
        log_hastings_ratio_denominator,
    )

    accepted = jr.bernoulli(acceptance_key, acceptance_probability)
    if verbosity > 1:
        debug.print("-" * 50)
        debug.print("Structure move accepted: {}", accepted)
        debug.print("Acceptance Probability: {}", acceptance_probability)
        debug.print("-" * 50)

    new_particle_state = lax.cond(
        accepted,
        lambda: proposed_particle_state,
        lambda: particle_state,
    )

    return new_particle_state, accepted


@partial(
    jit,
    static_argnames=(
        "kernel_prior",
        "data",
        "fix_noise",
        "verbosity",
    ),
)
def particle_state_proposal(
    key: PRNGKeyArray,
    particle_state: nnx.State,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    data: Dataset,
    fix_noise: bool,
    p_detach_attach: ScalarFloat = 0.5,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarFloat]:
    """
    Propose a new particle state by proposing a new kernel structure
    and noise variance, and calculating the contribution to the
    acceptance probability.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    particle_state : nnx.State
        The current particle state.
    kernel_prior : KernelPrior
        The kernel prior.
    noise_prior : Distribution
        The noise variance prior.
    data : Dataset
        The data.
    fix_noise : bool
        Whether to fix the noise variance. In case
        of fixed noise, the noise variance is not updated.
    p_detach_attach : ScalarFloat, optional
        The probability of performing a detach-attach move
        (vs a subtree-replace move), by default 0.5.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information
        is printed if verbosity > 1.

    Returns
    -------
    nnx.State
        The proposed particle state.
    ScalarFloat
        The log acceptance probability numerator.
    """

    # get necessary keys
    key, structure_key, noise_key = jr.split(key, 3)

    # unpack particle state
    kernel_state = particle_state.kernel
    noise_variance = jnp.array(particle_state.noise_variance.value)  # type: ignore

    # propose new kernel structure
    proposed_kernel_state, detach_attach_log_prob, performed_detach_attach = (
        kernel_structure_proposal(
            structure_key,
            kernel_state,
            kernel_prior,
            p_detach_attach,
            verbosity=verbosity,
        )
    )

    proposed_kernel = kernel_prior.reconstruct_kernel(proposed_kernel_state)

    # calculate gram matrix for new kernel
    kernel_gram = proposed_kernel._gram_train(data.x)

    # propose new noise variance
    proposed_noise_variance, log_prob_noise = lax.cond(
        fix_noise,
        lambda key: (noise_variance, 0.0),  # return noise variance as is
        lambda key: noise_variance_proposal(
            key,
            kernel_gram,
            noise_variance,
            data,
        ),
        noise_key,
    )

    # update particle state
    new_par_state = deepcopy(particle_state)
    new_par_state.kernel = proposed_kernel_state  # type: ignore
    new_par_state.noise_variance = (  # type: ignore
        particle_state.noise_variance.replace(  # type: ignore
            proposed_noise_variance,
        )
    )

    # calculate acceptance probability terms
    log_tree_size = jnp.log(proposed_kernel.node_sizes[0])
    marginal_log_likelihood = calculate_marginal_log_likelihood(
        kernel_gram,
        proposed_noise_variance,
        data,
    )
    log_prob_noise_prior = lax.cond(
        fix_noise,
        lambda x: jnp.array(0.0),
        lambda x: jnp.asarray(
            noise_prior.log_prob(x),
            dtype=data.x.dtype,
        ).squeeze(),
        proposed_noise_variance,
    )

    # calculate acceptance probability contribution,
    # (eq 22. and Preposition 2 in Saad2023)
    log_acceptance_prob_numerator = jnp.array(
        [
            marginal_log_likelihood,
            -log_tree_size,
            log_prob_noise_prior,
            -log_prob_noise,
            detach_attach_log_prob,
        ]
    ).sum()

    if verbosity > 1:
        lax.cond(
            performed_detach_attach,
            lambda: debug.print("Move Type: Detach-attach"),
            lambda: debug.print("Move Type: Subtree-replace"),
        )
        debug.print("-" * 50)
        debug.print("Proposal Terms:")
        debug.print("Marginal log likelihood: {}", marginal_log_likelihood)
        debug.print("Log (tree size^-1): {}", -log_tree_size)
        debug.print("Log noise prior probability: {}", log_prob_noise_prior)
        debug.print("Log (probability of noise posterior^-1): {}", -log_prob_noise)
        debug.print("Log detach-attach probability: {}", detach_attach_log_prob)
        debug.print("Sum of terms: {}", log_acceptance_prob_numerator)
        debug.print("-" * 50)

    return new_par_state, log_acceptance_prob_numerator


@partial(jit, static_argnames=("kernel_prior", "verbosity", "p_detach_attach"))
def kernel_structure_proposal(
    key: PRNGKeyArray,
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    p_detach_attach: ScalarFloat = 0.5,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarFloat, ScalarBool]:
    """
    Perform a move on the kernel structure,
    either detach-attach or subtree-replace.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    kernel_state : nnx.State
        The original kernel state.
    kernel_prior : KernelPrior
        The kernel prior.
    p_detach_attach : ScalarFloat, optional
        The probability of performing a detach-attach move
        (vs a subtree-replace move), by default 0.5.
    verbosity : int, optional
        The verbosity level, by default 0. Debug information
        for the detach-attach and subtree-replace moves
        are printed if verbosity > 2.

    Returns
    -------
    nnx.State
        The proposed kernel state.
    ScalarFloat
        The log probability associated with the detach-attach move
        (0.0 if a subtree-replace move is performed).
    """

    key, detach_attach_key, move_key = jr.split(key, 3)

    # choose move to perform (detach-attach or subtree-replace), with
    # detach-attach move being impossible if max_depth == 1
    p_detach_attach = jnp.where(
        kernel_prior.kernel_structure_prior.max_depth == 1,
        jnp.array(0.0),
        p_detach_attach,
    )
    perform_detach_attach = jr.bernoulli(
        detach_attach_key,
        p_detach_attach,
    )

    # perform structure move
    proposed_kernel_state, detach_attach_log_prob = lax.cond(
        perform_detach_attach,
        lambda key: detach_attach_move(
            key,
            kernel_state,
            kernel_prior,
            verbosity=verbosity,
        ),
        lambda key: (
            subtree_replace_move(
                key,
                kernel_state,
                kernel_prior,
                verbosity=verbosity,
            ),
            jnp.array(0.0),  # no detach-attach log prob if subtree-replace,
        ),  # (Preposition 1&2 in Saad2023)
        move_key,
    )

    return proposed_kernel_state, detach_attach_log_prob, perform_detach_attach
