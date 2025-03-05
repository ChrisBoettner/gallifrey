from functools import partial

import beartype.typing as tp
import jax.random as jr
from flax import nnx
from jax import jit, lax
from jax import numpy as jnp
from jaxtyping import Int, PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.inference.parameter_rejuvenation import rejuvenate_particle_parameters
from gallifrey.kernels.prior import KernelPrior
from gallifrey.moves.particle_move import structure_move
from gallifrey.utils.typing import ScalarBool, ScalarInt


@partial(
    jit,
    static_argnames=(
        "data",
        "kernel_prior",
        "noise_prior",
        "n_mcmc",
        "n_hmc",
        "fix_noise",
        "hmc_sampler_factory",
        "verbosity",
    ),
)
def rejuvenate_particle(
    key: PRNGKeyArray,
    particle_state: nnx.State,
    data: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    n_mcmc: ScalarInt,
    n_hmc: ScalarInt,
    fix_noise: bool,
    hmc_sampler_factory: tp.Callable,
    verbosity: int = 0,
) -> tuple[nnx.State, nnx.State, ScalarInt, ScalarInt]:
    """
    Rejuvenates a particle's state using a structure MCMC move followed by
    parameter HMC moves.

    This function performs a sequence of rejuvenation steps for a
    single particle state. It iterates `n_mcmc` times, applying the
    `rejuvenation_step` function in each iteration.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for the MCMC sampling.
    particle_state : nnx.State
        The current state of the particle to be rejuvenated.
    data : Dataset
        The dataset object containing the input and output data.
    kernel_prior : KernelPrior
        The kernel prior for the kernel structure.
    noise_prior : Distribution
        The prior distribution for the noise variance.
    n_mcmc : ScalarInt
        The number of MCMC iterations (rejuvenation steps) to perform.
    n_hmc : ScalarInt
        The number of Hamiltonian Monte Carlo (HMC) steps to perform within each
        parameter rejuvenation step (inside `rejuvenation_step`).
    fix_noise : bool
        A boolean flag indicating whether the noise variance is fixed or trainable.
    hmc_sampler_factory : tp.Callable
        A factory function that creates an HMC sampler.
    verbosity : int
        The verbosity level. Debug information is printed if `verbosity > 0`.

    Returns
    -------
    nnx.State
        The final rejuvenated particle state after `n_mcmc` iterations.
    nnx.State
        The history of the rejuvenation steps after every MCMC iteration.
    ScalarInt
        The total number of MCMC steps accepted during the rejuvenation process.
    ScalarInt
        The total number of HMC steps accepted during the rejuvenation process.
    """

    def scan_func(
        loop_state: tuple[nnx.State, Int[jnp.ndarray, ""], Int[jnp.ndarray, ""]],
        key: PRNGKeyArray,
    ) -> tuple[tuple[nnx.State, Int[jnp.ndarray, ""], Int[jnp.ndarray, ""]], nnx.State]:
        """
        A wrapper function for a single `rejuvenation_step` to be
        passed to the scan function.
        """
        particle_state, accepted_mcmc, accepted_hmc = loop_state

        new_state, mcmc_accepted, n_hmc_accepted = rejuvenation_step(
            key,
            particle_state,
            data,
            kernel_prior,
            noise_prior,
            n_hmc,
            fix_noise,
            hmc_sampler_factory,
            verbosity,
        )
        accepted_mcmc = accepted_mcmc + mcmc_accepted
        accepted_hmc = accepted_hmc + n_hmc_accepted

        return (new_state, accepted_mcmc, accepted_hmc), new_state

    final_loop_state, history = lax.scan(
        scan_func,
        (particle_state, jnp.array(0), jnp.array(0)),
        jr.split(key, int(n_mcmc)),
    )

    final_state, accepted_mcmc, accepted_hmc = final_loop_state

    return final_state, history, accepted_mcmc, accepted_hmc


@partial(
    jit,
    static_argnames=(
        "data",
        "kernel_prior",
        "noise_prior",
        "n_hmc",
        "fix_noise",
        "hmc_sampler_factory",
        "verbosity",
    ),
)
def rejuvenation_step(
    key: PRNGKeyArray,
    particle_state: nnx.State,
    data: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    n_hmc: ScalarInt,
    fix_noise: bool,
    hmc_sampler_factory: tp.Callable,
    verbosity: int = 0,
) -> tuple[nnx.State, ScalarBool, ScalarInt]:
    """
    Performs a single rejuvenation step for a
    article state.

    This function implements a single step of the rejuvenation process.
    It consists of two main moves:
        1. Structure Move: Proposes a new kernel structure based on the `kernel_prior`.
        2. Parameter Rejuvenation: If the structure move is accepted, rejuvenates
        the kernel parameters and potentially the noise variance using
        Hamiltonian Monte Carlo (HMC).

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for the rejuvenation step.
    particle_state : nnx.State
        The current state of the particle to be rejuvenated.
    data : Dataset
        The dataset object containing the input and output data.
    kernel_prior : KernelPrior
        The kernel prior for the kernel structure.
    noise_prior : Distribution
        The prior distribution for the noise variance.
    n_hmc : ScalarInt
        The number of Hamiltonian Monte Carlo (HMC) steps to perform during
        parameter rejuvenation.
    fix_noise : bool
        A boolean flag indicating whether the noise variance is fixed or trainable.
    hmc_sampler_factory : tp.Callable
        A factory function that creates an HMC sampler.
    verbosity : int
        The verbosity level. Debug information of the structure move
        is printed if `verbosity > 1`.

    Returns
    -------
    nnx.State
        The rejuvenated particle state after performing the structure move
        and (potentially) parameter rejuvenation.
    ScalarBool
        A boolean flag indicating whether the structure move was accepted.
    ScalarInt
        The number of HMC steps accepted during the parameter rejuvenation.
    """

    key, structure_key, hmc_key = jr.split(key, 3)

    new_particle_state, accepted = structure_move(
        structure_key,
        particle_state,
        kernel_prior,
        noise_prior,
        data,
        fix_noise=fix_noise,
        verbosity=verbosity,
    )

    new_particle_state, n_hmc_accepted = lax.cond(
        accepted,
        lambda key: rejuvenate_particle_parameters(
            key,
            new_particle_state,
            data,
            kernel_prior,
            noise_prior,
            n_hmc,
            hmc_sampler_factory,
        ),
        lambda key: (new_particle_state, jnp.array(0)),
        hmc_key,
    )

    return new_particle_state, jnp.array(accepted, dtype=jnp.bool), n_hmc_accepted
