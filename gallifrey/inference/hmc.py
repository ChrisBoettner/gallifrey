import beartype.typing as tp
import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.mcmc.hmc import HMCState
from flax import nnx
from jax import jit
from jaxtyping import PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.kernels.prior import KernelPrior
from gallifrey.parameter import ParticleParameter
from gallifrey.particles import transform_particle_parameters
from gallifrey.utils.probability_calculations import calculate_marginal_log_likelihood
from gallifrey.utils.typing import ScalarFloat, ScalarInt


def get_hmc_objective(
    particle_state: nnx.State,
    data: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
) -> tp.Callable:
    """
    Wrapper function to create the HMC objective function.

    The HMC objective function is the sum of the marginal log likelihood
    and the log prior of the kernel and noise parameters.

    The wrapper splits the particle state into a parameter state (
    used for sampling) and a static state.
    The resulting objective function takes the particle parameter
    state in the unconstrained space, transforms it to the constrained
    space, and returns the value of the objective function.

    Parameters
    ----------
    particle_state : nnx.State
        The particle state.
    data : Dataset
        A dataset object containing the input x and output y.
    kernel_prior : KernelPrior
        The kernel prior object.
    noise_prior : Distribution
        The noise prior distribution.

    Returns
    -------
    tp.Callable
        The HMC objective function, which takes the unconstrained
        particle parameter state and returns the value of the objective
        function.
    """

    _, *particle_static_state = particle_state.split(ParticleParameter, ...)

    @jit
    def hmc_objective(
        unconstrained_particle_parameter_state: nnx.State,
    ) -> ScalarFloat:
        """HMC objective function"""

        unconstrained_particle_state = nnx.State.merge(
            unconstrained_particle_parameter_state,
            *particle_static_state,
        )

        constrained_particle_state = transform_particle_parameters(
            unconstrained_particle_state,
            kernel_prior,
            inverse=False,
        )

        kernel_state = constrained_particle_state.kernel
        kernel = kernel_prior.reconstruct_kernel(kernel_state)

        # calcuate marginal log likelihood
        kernel_gram = kernel._gram_train(data.x)
        noise_variance = jnp.array(
            constrained_particle_state.noise_variance.value,
        )  # type: ignore
        mll = calculate_marginal_log_likelihood(
            kernel_gram,
            noise_variance,
            data,
        )

        # calculate priors
        log_prob_kernel_parameter_prior = kernel_prior.parameter_prior.log_prob(
            kernel_state
        )
        log_prob_noise_parameter_prior = jnp.asarray(
            noise_prior.log_prob(noise_variance),
            dtype=data.x.dtype,
        ).squeeze()

        return mll + log_prob_kernel_parameter_prior + log_prob_noise_parameter_prior

    return hmc_objective


def hmc_inference_loop(
    key: PRNGKeyArray,
    hmc_kernel: tp.Callable,
    initial_state: HMCState,
    n_hmc: ScalarInt,
    transform: tp.Callable = lambda state, info: (state, info),
) -> tuple[HMCState, tp.Any]:
    """
    Run an HMC inference loop.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    hmc_kernel : tp.Callable
        The HMC kernel (e.g. HMC, NUTS), derived
        from one of blackjax's HMC sampling algorithms.
    initial_state : HMCState
        The initial state of the HMC sampler.
    n_hmc : ScalarInt
        The number of HMC steps.
    transform : tp.Callable, optional
        A transformation of the trace of states (and info) to be returned.
        This is useful for computing determinstic variables, or
        returning a subset of the states. By default, the states are
        returned as a tuple (state, info).

    Returns
    -------
    HMCState
        The final state.
    tp.Any
        The history of states. The output type depends on the
        transform function. By default, the states are a tuple of
        the state and the info.
    """

    @jit
    def one_step(state: HMCState, key: PRNGKeyArray) -> tuple[HMCState, tp.Any]:
        new_state, info = hmc_kernel(key, state)
        return new_state, transform(new_state, info)

    keys = jr.split(key, int(n_hmc))
    final_state, history = jax.lax.scan(
        one_step,
        initial_state,
        keys,
    )

    return final_state, history


def create_hmc_sampler_factory(
    hmc_config: dict[str, float],
    num_parameter: int,
) -> tp.Callable:
    """
    Create a Blackjax HMC sampler factory.
    The returned function takes an objective function and returns
    a Blackjax HMC sampler.

    This wrapper is used to apply the HMC config to the HMC sampler.

    The inverse mass matrix is diagonal with ones, and scaled by the
    inverse mass matrix scaling factor in the config.

    Parameters
    ----------
    hmc_config : dict[str, float]
        A dictionary containing the HMC config parameters, must
        contain the keys:
        - "step_size"
        - "inv_mass_matrix_scaling"
        - "num_integration_steps"
    num_parameter : int
        The (maximum) number of parameters in the model.

    Returns
    -------
    tp.Callable
        A function that takes an objective function and returns
        a Blackjax HMC sampler.
    """

    def hmc_sampler_factory(
        objective_function: tp.Callable,
    ) -> blackjax.SamplingAlgorithm:

        step_size = hmc_config["step_size"]
        inv_mass_matrix = hmc_config["inv_mass_matrix_scaling"] * jnp.diag(
            jnp.ones(num_parameter)
        )
        num_integration_steps = hmc_config["num_integration_steps"]

        hmc_sampler = blackjax.hmc(
            objective_function,
            step_size,
            inv_mass_matrix,
            num_integration_steps,
        )

        return hmc_sampler

    return hmc_sampler_factory
