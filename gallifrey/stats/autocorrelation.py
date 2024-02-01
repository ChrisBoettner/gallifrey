import jax.numpy as jnp
from beartype.typing import Iterable, Optional
from jax.numpy.fft import fft, ifft
from jaxtyping import Array
from tensorflow_probability.substrates.jax.distributions import Normal


def acf(data: Array, axis: int = 0, max_lag: Optional[int] = None) -> Array:
    """
    Calculate the autocorrelation function of some data up to some
    maximum lag.
    Returns lag 1,..., max_lag.

    Parameters
    ----------
    data : Array
        The data, can be 1D or 2D.
    axis : int, optional
        The axis along which to calculate the ACF, by default 0.
    max_lag : Optional[int], optional
        The maximum lag to consider. By default None, which
        uses min(10, data.shape[axis] // 5).

    Returns
    -------
    Array
        The autocorrelation function along the specified axis.
    """

    data = data.reshape(-1, 1) if data.ndim == 1 else data
    max_lag = int(min(10, data.shape[axis] // 5)) if max_lag is None else max_lag

    # move the specified axis to the front
    data = jnp.swapaxes(data, axis, 0) if axis != 0 else data

    # centre the data
    centred_data = data - jnp.mean(data, axis=0)

    # pad the data to avoid wraparound effect
    padded_data = jnp.pad(
        centred_data,
        ((0, centred_data.shape[0]), (0, 0)),
        mode="constant",
    )

    # compute power spectrum
    fft_series = fft(padded_data, axis=0)
    power_spectrum = jnp.abs(fft_series) ** 2

    # compute the autocorrelation function
    autocorr_func = ifft(power_spectrum, axis=0).real

    # normalize by autocorrelation at lag 0 and select up to max_lag
    normalised_acf = autocorr_func[: max_lag + 1] / autocorr_func[0]

    # remove lag 0
    normalised_acf = normalised_acf[1:]

    # move axis back
    normalised_acf = (
        jnp.swapaxes(normalised_acf, 0, axis) if axis != 0 else normalised_acf
    )
    return normalised_acf


def rho_squared(
    data: Array,
    axis: int = 0,
    max_lag: Optional[int] = None,
) -> Array:
    """
    Calculate the lag-n autocorrelation for lag 1 to max_lag, and sum the squared
    results, then normalise it.
    The final value should be near 1, if the data is uncorrelated. Use ;;; to
    calculate the credible regions for a standard normal baseline.

    Parameters
    ----------
    data : Array
        The data, can be 1D or 2D.
    axis : int, optional
        The axis along which to calculate the statistic, by default 0.
    max_lag : Optional[int], optional
        The maximum lag to consider. By default None, which
        uses min(10, data.shape[axis] // 5).

    Returns
    -------
    Array
        The value of the autocorrelation test statistic.
    """

    data = data.reshape(-1, 1) if data.ndim == 1 else data
    max_lag = int(min(10, data.shape[axis] // 5)) if max_lag is None else max_lag

    # move the specified axis to the front
    data = jnp.swapaxes(data, axis, 0) if axis != 0 else data

    acf_values = acf(data, max_lag=max_lag)
    return jnp.sum(acf_values**2, axis=0) * data.shape[axis] / max_lag


def rho_squared_region(
    rng_key: Array,
    n_data: int,
    percentiles: Iterable | float = [2.5, 97.5],
    n_samples: int = 10000,
    variance: float = 1,
    max_lag: Optional[int] = None,
) -> Array:
    """
    Calculate the expected percentile region for the rho squared statistic
    to fall in. This is done brute-force, by calculating n_samples white
    noise time series with n_data points each, calculating the statistic
    for each one and calculating the percentiles based on that sample.

    Parameters
    ----------
    rng_key : Array
        The rng key used to generate the samples
    n_data : int
        The number of datapoints created per sample.
    percentiles : Iterable | float, optional
        The percentiles to return, by default [0.025, 0.975].
    n_samples : int, optional
        The number of samples to generate, by default 10000.
    variance : float, optional
        The variance of the Normal distribution. By default 1,
        which would be expected for white noise.
    max_lag : Optional[int], optional
        The maximum lag up to which the autocorrelation function
        is calculated. By default None, which
        uses min(10, data.shape[axis] // 5).

    Returns
    -------
    Array
        The percentiles for the rho_squared credible region.
    """

    max_lag = int(min(10, n_data // 5)) if max_lag is None else max_lag

    percentiles = jnp.atleast_1d(jnp.asarray(percentiles))
    assert isinstance(percentiles, Array)  # type: ignore

    samples = jnp.array(
        Normal(loc=0, scale=jnp.sqrt(variance)).sample(
            seed=rng_key, sample_shape=(n_data, n_samples)
        )
    )

    rho_squared_statistic = rho_squared(samples, max_lag=max_lag)

    region = jnp.percentile(rho_squared_statistic, percentiles)

    return region
