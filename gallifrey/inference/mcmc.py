import blackjax
import jax
import jax.numpy as jnp
from beartype.typing import Callable, NamedTuple
from blackjax.base import SamplingAlgorithm
from blackjax.progress_bar import progress_bar_scan
from blackjax.types import ArrayLikeTree
from jax import jit
from jaxtyping import Array


def nuts_warmup(
    rng_key: Array,
    log_probability: Callable,
    initial_position: ArrayLikeTree,
    num_steps: int = 500,
    target_acceptance_rate: float = 0.65,
    progress_bar: bool = True,
) -> tuple:
    """Runs burn-in for NUTS mcmc
    sampling. Also automatically
    adapts sampler parameter.

    Parameters
    ----------
    rng_key : Array
        A random key for the sampling.
    log_probability : Callable
        The function to be sampled.
    initial_position : ArrayLikeTree
        Initial position of the sampler.
    num_steps : int, optional
        The number of steps to run the burn in for,
        by default 500
    target_acceptance_rate : float, optional
        The target acceptance rate of mcmc proposals,
        by default 0.65
    progressbar : bool, optional
        Whether to display a progress bar, by default
        True

    Returns
    -------
    tuple[ArrayTree, dict]
        Returns final state of burn-in, which
        can be used as initial position for sampling
        run, and optimised parameter for sampler.
    """
    warmup = blackjax.window_adaptation(
        blackjax.nuts,  # type: ignore
        log_probability,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
    )
    (state, parameters), _ = warmup.run(
        rng_key,
        initial_position,
        num_steps=num_steps,  # type: ignore
    )
    return state, parameters


def inference_algorithm(
    rng_key: Array,
    initial_state_or_position: ArrayLikeTree | NamedTuple,
    inference_algorithm: SamplingAlgorithm,
    num_steps: int,
    progress_bar: bool = False,
) -> tuple:
    """The inference loop for the Blackjax sampling,
    adapted from the Blackjax code.

    Parameters
    ----------
    rng_key : Array
        jax random key
    initial_state_or_position : ArrayLikeTree | NamedTuple
        The initial position for the sampling.
    inference_algorithm : SamplingAlgorithm
        The Blackjax algorithm used for sampling.
    num_steps : int
        The number of steps for which to sample
    progress_bar : bool, optional
        Wether the progressbar should be shown, by
        default False

    Returns
    -------
    tuple
        Tuple of final_state, state_history, and
        info_history
    """
    init_key, sample_key = jax.random.split(rng_key, 2)
    try:
        initial_state = inference_algorithm.init(
            initial_state_or_position,
            init_key,  # type: ignore
        )
    except (TypeError, ValueError, AttributeError):
        # We assume initial_state is already in the right format.
        initial_state = initial_state_or_position

    keys = jax.random.split(sample_key, num_steps)

    @jit
    def _one_step(state: NamedTuple, xs: tuple) -> tuple:
        _, rng_key = xs
        state, info = inference_algorithm.step(rng_key, state)
        return state, (state, info)

    if progress_bar:
        one_step = progress_bar_scan(num_steps)(_one_step)
    else:
        one_step = _one_step

    xs = (jnp.arange(num_steps), keys)
    final_state, (state_history, info_history) = jax.lax.scan(
        one_step, initial_state, xs  # type: ignore
    )
    return final_state, state_history, info_history


def run_mcmc(
    rng_key: Array,
    log_probability: Callable,
    parameters: dict,
    initial_positions: ArrayLikeTree | NamedTuple,
    num_steps: int = 500,
) -> tuple:
    """Run the (parallel) NUTS mcmc sampling.

    Parameters
    ----------
    rng_key : Array
        A rng key for the sampling
    log_probability : Callable
        The function to be sampled.
    parameters : dict
        The parameter for the NUTS sampler,
        usually determined using nuts_warmup
    initial_positions : ArrayLikeTree | NamedTuple
        The initial positions to be passed
        to the sampler, number of initial positions
        determines how many parallel chains are run.
        At most as many as cores are available (on CPU).
    num_steps : int, optional
        _description_, by default 500

    Returns
    -------
    tuple
        Tuple of final_state, state_history, and
        info_history
    """
    nuts = blackjax.nuts(log_probability, **parameters)

    initial_states = jax.vmap(nuts.init, in_axes=(0))(initial_positions)  # type: ignore

    inference_loop_multiple_chains = jax.pmap(
        inference_algorithm,
        in_axes=(0, 0, None, None, None),  # type: ignore
        static_broadcasted_argnums=(2, 3, 4),
    )

    sample_keys = jax.random.split(rng_key, len(initial_states[1]))

    pmap_states = inference_loop_multiple_chains(
        sample_keys,
        initial_states,
        nuts,
        num_steps,
        False,
    )
    _ = pmap_states[1].position["lc_parameter"].block_until_ready()

    final_state, state_history, info_history = pmap_states
    return final_state, state_history, info_history


def gelman_rubin_diagnostic(chains: Array) -> Array:
    """Compute the Gelman-Rubin diagnostic for a set of MCMC
    chains. Values close to 1 indicate the chains are converged.

    Parameters
    ----------
    chains : array
        An array of shape (num_chains, num_samples, num_parameters)
        representing the sampled chains.

    Returns
    -------
    Array
        Gelman-Rubin diagnostic for each parameter.
    """
    # Number of chains and number of samples per chain
    num_samples = chains.shape[1]

    # Calculate the within-chain variance
    # Mean of each chain
    chain_means = jnp.mean(chains, axis=1)
    # Variance within each chain
    within_chain_var = jnp.var(chains, axis=1, ddof=1)
    # Average of the within-chain variances
    W = jnp.mean(within_chain_var, axis=0)

    # Calculate the between-chain variance
    # Variance of the chain means
    B_over_n = jnp.var(chain_means, axis=0, ddof=1)

    # Estimate of marginal posterior variance
    var_plus = ((num_samples - 1) / num_samples) * W + B_over_n

    R_hat = var_plus / W
    return R_hat
