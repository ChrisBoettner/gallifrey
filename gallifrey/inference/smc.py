from copy import deepcopy
from functools import partial

import beartype.typing as tp
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from blackjax.smc.resampling import stratified
from flax import nnx
from jax import jit, lax, pmap, vmap
from jax.scipy.special import logsumexp
from jaxtyping import Float, PRNGKeyArray
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.data import Dataset
from gallifrey.inference.rejuvenation import rejuvenate_particle
from gallifrey.inference.state import GPState
from gallifrey.kernels.prior import KernelPrior
from gallifrey.utils.probability_calculations import calculate_marginal_log_likelihood
from gallifrey.utils.typing import ScalarFloat, ScalarInt


def calculate_mll_wrapper(
    particle_state: nnx.State,
    data: Dataset,
    kernel_prior: KernelPrior,
) -> ScalarFloat:
    """
    Wapper around calculate_marginal_log_likelihood, which
    calculates the MLL from a particle state.

    Parameters
    ----------
    particle_state : nnx.State
        The particle state.
    data : Dataset
        The dataset object, containing the input x and output y.
    kernel_prior : KernelPrior
        The kernel prior, used to reconstruct the kernel.

    Returns
    -------
    ScalarFloat
        The marginal log likelihood for the given particle state and
        subset of the data.

    """

    kernel = kernel_prior.reconstruct_kernel(particle_state.kernel)
    kernel_gram = kernel._gram_train(data.x)

    noise_variance = jnp.array(particle_state.noise_variance.value)  # type: ignore

    mll = calculate_marginal_log_likelihood(
        kernel_gram,
        noise_variance,
        data,
    )
    return mll


@partial(jit, static_argnames=("data", "kernel_prior"))
def reweight_particles(
    smc_state: GPState,
    data: Dataset,
    kernel_prior: KernelPrior,
) -> Float[jnp.ndarray, " N"]:
    """
    Reweights particles based on the marginal log likelihood
    calculated with new data points.

    This function updates the log weights of the particles in the SMC state by
    calculating the marginal log likelihood for each particle using the new subset
    vs the previous subset, taking the difference in MLL for each particle and adding
    it to the existing log weights, in essence:

        w_t = w_{t-1} * (mll_t / mll_{t-1}) (but in log space, so addition instead)

    See Saad2023, algorithm 1 line 5.

    The weights are returned normalised.

    Parameters
    ----------
    smc_state : GPState
        The SMC state after the previous round.
    data : Dataset
        The dataset containing the data at the current time point.
    kernel_prior : KernelPrior
        The kernel prior object, used to construct the kernel.

    Returns
    -------
    Float[jnp.ndarray, " N"]
        The updated (normalised) log weights for the particles.
    """

    new_mlls = vmap(calculate_mll_wrapper, in_axes=(0, None, None))(
        smc_state.particle_states,  # type: ignore
        data,
        kernel_prior,
    )

    mll_ratio = new_mlls - smc_state.marginal_log_likelihoods  # type: ignore

    unnorm_log_weights: jnp.ndarray = smc_state.log_weights + mll_ratio  # type: ignore
    log_weights = unnorm_log_weights - logsumexp(unnorm_log_weights)
    return log_weights


def calculate_effective_sample_size(
    log_weights: Float[jnp.ndarray, " N"]
) -> ScalarFloat:
    """
    Calculates the effective sample size (ESS) of the particles.

    The effective sample size is a measure of how diverse the particle weights are,
    and is used to determine whether resampling is needed. If the ESS is low, it means
    that many particles have similar weights, and resampling is needed to avoid
    degeneracy.

    The ESS is calculated as:

        ESS = 1 / sum(w^2)

    where w is the normalised(!) weight of each particle.

    Parameters
    ----------
    log_weights : Float[jnp.ndarray, " N"]
        The log weights of the particles.

    Returns
    -------
    ScalarFloat
        The effective sample size of the particles.

    """

    weights = jnp.exp(log_weights)
    ess = 1 / jnp.sum(weights**2)
    return ess


@partial(jit, static_argnames=("num_particles", "resampling_func"))
def resample_particles(
    key: PRNGKeyArray,
    particle_states: nnx.State,
    log_weights: Float[jnp.ndarray, " N"],
    num_particles: ScalarInt,
    resampling_func: tp.Callable,
) -> tuple[nnx.State, Float[jnp.ndarray, " N"]]:
    """
    Resamples particles based on the log weights.

    This function resamples the particles based on the log weights
    using a resampling function, which takes the log weights and
    the number of particles and returns the indices of the resampled
    particles.

    The resampling function must have the signature:
        resampling_func(key, weights, num_particles) -> jnp.ndarray

    Note that the actual weights must be passed to the resampling function
    (not the log weights), and that they need to be normalised.

    We use sampling functions from blackjax. By default we use stratified
    resampling, but systematic, multinomial, and residual resampling are also
    available.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key for the resampling function.
    particle_states : nnx.State
        The current particle states.
    log_weights : Float[jnp.ndarray, " N"]
        The log weights of the particles.
    num_particles : ScalarInt
        The number of particles (should be the same as the length of the log weights,
        and the number of particles in the particle_states).
    resampling_func : tp.Callable
        The resampling function to use.

    Returns
    -------
    nnx.State
        The resampled particle states.
    Float[jnp.ndarray, " N"]
        The normalised log weights of the resampled particles.
        (i.e. log(1/N) for each particle, see Saad2023, algorithm 1 line 9).
    """

    # sample indices based on weights
    sampled_indices = resampling_func(key, jnp.exp(log_weights), num_particles)

    # update particle states based on sampled indices
    resampled_state = jtu.tree_map(lambda leaf: leaf[sampled_indices], particle_states)

    resampled_log_weights = jnp.log(jnp.ones(num_particles) / num_particles)

    return resampled_state, resampled_log_weights


def smc_round(
    smc_state: GPState,
    num_particles: ScalarInt,
    n_mcmc: ScalarInt,
    n_hmc: ScalarInt,
    data_subset: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    fix_noise: bool,
    hmc_sampler_factory: tp.Callable,
    resampling_func: tp.Callable,
    ess_threshold: float,
    total_data_points: ScalarInt,
    verbosity: ScalarInt = 0,
) -> GPState:
    """
    Performs one round of Sequential Monte Carlo (SMC).

    This function executes a single SMC round, which includes reweighting particles
    based on new data, resampling particles if the effective sample size is below
    a threshold, and rejuvenating particles using MCMC and HMC steps.

    Parameters
    ----------
    smc_state : GPState
        The current SMC state.
    num_particles : ScalarInt
        The number of particles.
    n_mcmc : ScalarInt
        The number of MCMC steps to perform during rejuvenation.
    n_hmc : ScalarInt
        The number of HMC steps within each MCMC step during rejuvenation.
    data_subset : Dataset
        The data to be considered for this SMC round.
    kernel_prior : KernelPrior
        The kernel prior object.
    noise_prior : Distribution
        The prior distribution for the noise parameter.
    fix_noise : bool
        Whether to fix the noise parameter during rejuvenation.
    hmc_sampler_factory : tp.Callable
        A factory function that creates an HMC sampler.
    resampling_func : tp.Callable
        The resampling function to use.
    ess_threshold : float
        The threshold for the effective sample size (ESS) below which
        resampling is performed.
    total_data_points : ScalarInt
        The total number of data points in the dataset.
    verbosity : ScalarInt, optional
        The verbosity level for the rejuvenation process. Debug information
        is printed if verbosity > 0.

    Returns
    -------
    SMState
        The updated SMC state after the round.

    """

    key, resample_key, rejuvenate_key, state_key = jr.split(
        smc_state.key,  # type: ignore
        4,
    )

    # reweight
    new_log_weights = reweight_particles(
        smc_state,
        data_subset,
        kernel_prior,
    )
    if verbosity > 0:
        print(f"Weights: {jnp.exp(new_log_weights)}")

    # get number of data points for this round
    num_data_points = data_subset.x.shape[0]

    # maybe resample, depending on effective sample size (ESS)
    # (if resample, also reset log weights to uniform)
    normalised_ess = (
        calculate_effective_sample_size(new_log_weights)
        / smc_state.num_particles  # type: ignore
    )
    # do not resample if we are at last step
    do_resample = (normalised_ess < ess_threshold) & (
        num_data_points < total_data_points
    )

    resampled_particle_states, new_log_weights, resampled = lax.cond(
        do_resample,
        lambda key: (
            *resample_particles(
                resample_key,
                smc_state.particle_states,
                new_log_weights,
                num_particles,
                resampling_func,
            ),
            True,
        ),
        lambda key: (smc_state.particle_states, new_log_weights, False),
        key,
    )
    if verbosity > 0:
        print(
            f"Resampled: {resampled} (Normalised ESS: "
            f"{float(normalised_ess):.2f})"  # type: ignore
        )

    # rejuvenate
    @jit
    def wrapper(
        key: PRNGKeyArray,
        state: nnx.State,
    ) -> tuple[nnx.State, nnx.State, ScalarInt, ScalarInt]:
        """Wrapper around the rejuvenate_particle function using
        GPmodel attributes."""
        return rejuvenate_particle(
            key,
            state,
            data_subset,
            kernel_prior,
            noise_prior,
            n_mcmc=n_mcmc,
            n_hmc=n_hmc,
            fix_noise=fix_noise,
            hmc_sampler_factory=hmc_sampler_factory,
            verbosity=verbosity,
        )

    # run rejuvenation on all particles, parallelised
    final_state, _, accepted_mcmc, accepted_hmc = pmap(wrapper, in_axes=0)(
        jr.split(rejuvenate_key, num_particles),  # type: ignore
        resampled_particle_states,
    )

    if verbosity > 0:
        for i, acc_mcmc, acc_hmc in zip(
            range(smc_state.num_particles),  # type: ignore
            accepted_mcmc,
            accepted_hmc,
        ):
            print(
                f"Particle {i+1} | Accepted: MCMC[{acc_mcmc}/{n_mcmc}] "
                f" HMC[{acc_hmc}/{acc_mcmc*n_hmc}]"
            )
        print("=" * 50)

    # calculate annealing MLL
    new_mlls = jnp.array(
        jit(
            vmap(calculate_mll_wrapper, in_axes=(0, None, None)),
            static_argnames=("data", "kernel_prior"),
        )(
            final_state,
            data_subset,
            kernel_prior,
        )
    )

    return GPState(
        particle_states=final_state,
        log_weights=new_log_weights,
        marginal_log_likelihoods=new_mlls,
        num_particles=smc_state.num_particles,
        num_data_points=num_data_points,
        resampled=resampled,
        mcmc_accepted=accepted_mcmc,
        hmc_accepted=accepted_hmc,
        key=state_key,
    )


# @partial(
#     jit,
#     static_argnames=(
#         "annealing_schedule",
#         "num_particles",
#         "data",
#         "kernel_prior",
#         "noise_prior",
#         "fix_noise",
#         "hmc_sampler_factory",
#         "n_mcmc",
#         "n_hmc",
#         "resampling_func",
#         "verbosity",
#     ),
# )
def smc_loop(
    key: PRNGKeyArray,
    particle_states: nnx.State,
    annealing_schedule: tuple[int, ...],
    num_particles: int,
    data: Dataset,
    kernel_prior: KernelPrior,
    noise_prior: Distribution,
    fix_noise: bool,
    hmc_sampler_factory: tp.Callable,
    n_mcmc: ScalarInt,
    n_hmc: ScalarInt,
    resampling_func: tp.Callable = stratified,
    ess_threshold: float = 0.5,
    verbosity: ScalarInt = 0,
) -> tuple[GPState, list[GPState]]:
    """
    Runs the Sequential Monte Carlo (SMC) algorithm.

    The SMC loop runs a series of SMC rounds using a data annealing
    schedule, each round includes three steps:

    - Reweighting: Update particle weights based on new data.
    The new weight is the old weight times the ratio of the new
    marginal likelihood evaluated on the new data (so all data up
    to n_t, where n_t is the number of data points at time t in
    the annealing schedule) and the old marginal likelihood evaluated
    on the previous data (up to n_{t-1}). In essence, the new weight
    reflect how much better the particle fits the new (unseen) data
    compared to the data it was trained on.

    - Resampling: Adaptively resample particles based on the new weights.
    This step is performed if the effective sample size (ESS) is below a
    threshold (0.5 in this implementation). The ESS is a measure of how
    diverse the particle weights are, and if it is low, it means that many
    particles have similar weights, and resampling is needed to avoid
    degeneracy. There are various resampling strategies available, we
    default to stratified resampling.

    - Rejuvenation: Rejuvenate particles using MCMC and HMC steps. This
    step is performed on all particles, and consists of two main moves:
        1. Structure Move: Proposes a new kernel structure based on the
        `kernel_prior`.
        2. Parameter Rejuvenation: If the structure move is accepted,
        rejuvenates the kernel parameters and potentially the noise variance
        using Hamiltonian Monte Carlo (HMC).



    Parameters
    ----------
    key : PRNGKeyArray
        The random key used for sampling.
    particle_states : nnx.State
        The particle states at the start of the SMC loop.
    annealing_schedule : Int[jnp.ndarray, " T"]
        The annealing schedule, an array of integers representing
        the number of data points to use at each time point
        (e.g. jnp.array([10, 20, 30, 40, 50]) ).
    num_particles : int
        The number of particles (must match number of particles
        in the particle_states).
    data : Dataset
        The dataset object containing the input x and output y for
        all data points.
    kernel_prior : KernelPrior
        The kernel prior object, used to reconstruct the kernel, and sample
        new kernel structures.
    noise_prior : Distribution
        The prior distribution for the noise variance.
    fix_noise : bool
        Whether to fix the noise variance during rejuvenation, or sample it.
    hmc_sampler_factory : tp.Callable
        A factory function that creates an HMC sampler instance for the parameter
        rejuvenation step.
    n_mcmc : ScalarInt
        The number of MCMC steps to perform during rejuvenation.
    n_hmc : ScalarInt
        The number of HMC steps to perform during rejuvenation.
    resampling_func : tp.Callable, optional
        The resampling function to use, by default stratified.
    ess_threshold : float, optional
        The threshold for the effective sample size (ESS) below which
        resampling is performed, by default 0.5.
    verbosity : int, optional
        The verbosity level for the rejuvenation process. Debug information
        is printed if verbosity > 0.

    Returns
    -------
    GPState
        The final SMC state after the SMC loop.
    list[GPState]
        The full history of the SMC state at each time point in the
        annealing schedule.

    """

    def smc_round_wrapper(
        smc_state: GPState,
        points_to_use: int,
    ) -> GPState:
        """Wrapper around the smc_round function."""

        x_slice = lax.slice(data.x, (0,), (points_to_use,))
        y_slice = lax.slice(data.y, (0,), (points_to_use,))

        kernel_prior_round = deepcopy(kernel_prior)
        kernel_prior_round.num_datapoints = points_to_use

        data_subset = Dataset(x=x_slice, y=y_slice)

        return smc_round(
            smc_state,
            num_particles,
            n_mcmc,
            n_hmc,
            data_subset,
            kernel_prior_round,
            noise_prior,
            fix_noise,
            hmc_sampler_factory,
            resampling_func,
            ess_threshold,
            data.x.shape[0],
            verbosity,
        )

    # initialise SMC state
    initial_smc_state = GPState(
        particle_states=particle_states,
        log_weights=jnp.log(jnp.ones(num_particles) / num_particles),
        marginal_log_likelihoods=jnp.zeros(num_particles),
        num_particles=num_particles,
        num_data_points=jnp.array(0),
        resampled=False,
        mcmc_accepted=jnp.zeros(num_particles, dtype=int),
        hmc_accepted=jnp.zeros(num_particles, dtype=int),
        key=key,
    )

    history = []
    for i, points_to_use in enumerate(annealing_schedule):
        if verbosity > 0:
            print(
                f"Running SMC round [{i+1}/{len(annealing_schedule)}] "
                f"with [{points_to_use}/{len(data.x)}] data points."
            )

        final_smc_state = smc_round_wrapper(
            initial_smc_state,
            points_to_use,
        )
        initial_smc_state = final_smc_state
        history.append(final_smc_state)

    return final_smc_state, history
