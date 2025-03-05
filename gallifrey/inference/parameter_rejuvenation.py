from functools import partial

import beartype.typing as tp
import jax
import jax.random as jr
from flax import nnx
from jax import jit
from jaxtyping import PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.inference.hmc import get_hmc_objective, hmc_inference_loop
from gallifrey.kernels.prior import KernelPrior
from gallifrey.parameter import ParticleParameter
from gallifrey.particles import transform_particle_parameters
from gallifrey.utils.typing import ScalarInt


@partial(
    jit,
    static_argnames=(
        "data",
        "kernel_prior",
        "noise_prior",
        "n_hmc",
        "hmc_sampler_factory",
    ),
)
def rejuvenate_particle_parameters(
    key: PRNGKeyArray,
    particle_state: nnx.State,
    data: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    n_hmc: ScalarInt,
    hmc_sampler_factory: tp.Callable,
) -> tuple[nnx.State, ScalarInt]:
    """
    Rejuvenate GP parameters using HMC.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    particle_state : nnx.State
        The current particle state.
    data : Dataset
        The dataset object containing
        the input x and output y.
    kernel_prior : KernelPrior
        The kernel prior object.
    noise_prior : Distribution
        The noise prior distribution.
    n_hmc : ScalarInt
        The number of HMC steps.
    hmc_sampler_factory : tp.Callable
        The HMC sampler factory function,
        which takes an objective function and
        returns a Blackjax HMC sampler.

    Returns
    -------
    nnx.State
        The particle state with rejuvenated
        parameters.
    ScalarInt
        The number of accepted HMC steps.

    """

    # get HMC objective and sampler
    hmc_objective = get_hmc_objective(
        particle_state,
        data,
        kernel_prior,
        noise_prior,
    )
    hmc_sampler = hmc_sampler_factory(hmc_objective)

    # transform particle parameters to unconstrained space and
    # create initial HMC state
    unconstrained_particle_state = transform_particle_parameters(
        particle_state,
        kernel_prior,
        inverse=True,
    )
    unconstrained_parameter_state, *static_state = unconstrained_particle_state.split(
        ParticleParameter, ...
    )
    initial_hmc_state = hmc_sampler.init(unconstrained_parameter_state)

    # run the HMC inference loop
    key, sample_key = jr.split(key)
    final_parameter_state, other = hmc_inference_loop(
        sample_key,
        jax.jit(hmc_sampler.step),
        initial_hmc_state,
        n_hmc,
    )
    history, info = other
    # HMCInfo contains (maybe we could return more):
    # - momentum
    # - acceptance_rate
    # - is_accepted
    # - is_divergent
    # - energy
    # - proposal
    # - step_size
    # - num_integration_steps

    # put particle state back together and transform to constrained space
    new_particle_state = nnx.State.merge(
        final_parameter_state.position,  # type: ignore
        *static_state,
    )
    constrained_particle_state = transform_particle_parameters(
        new_particle_state,
        kernel_prior,
        inverse=False,
    )

    return constrained_particle_state, info.is_accepted.sum()
