import jax.numpy as jnp
import jax.random as jr
import numpy as np
from beartype.typing import Any
from jax import jit
from jaxoplanet import orbits
from jaxoplanet.light_curves import LimbDarkLightCurve
from jaxtyping import Array
from numpy.typing import NDArray
from scipy.stats import chi2

rng_key = jr.PRNGKey(36)
idx_key, noise_key = jr.split(rng_key, 2)


@jit
def example_transit_model(t: Array, params: list) -> Array:
    """Example transit model using jaxoplanet.

    Parameters
    ----------
    t : Array
        The time's at which to evaluate the light curve.
    params : list
        The light curve parameter (radius, u1, u2).

    Returns
    -------
    Array
        The example transit light curve.
    """
    orbit = orbits.keplerian.Body(
        period=15,
        radius=params[0],
        inclination=jnp.deg2rad(89),
        time_transit=0,
    )

    lc = LimbDarkLightCurve([params[1], params[2]]).light_curve(orbit, t=t)
    return lc


def example_lightcurve(
    noise_std: float = 0.001,
    correlated: bool = False,
    num_train: int = 150,
) -> dict:
    """Calculate a toy example light curve.

    Parameters
    ----------
    noise_std : float, optional
        White noise std, by
        default 0.001.
    phi : float, optional
        If True, create AR(1) noise with phi=0.8, rather
        than white noise. By default False.
    num_train : int, optional
        Number of tranings points (<1000), by
        default 150.

    Returns
    -------
    dict
        Dictonary containing the toy example.
    """
    example: dict[str, Any] = {}

    example["white_noise_std"] = noise_std
    if correlated:
        example["phi"] = 0.8
        example["name"] = "gpmodel_ar_noise"
    else:
        example["phi"] = 0
        example["name"] = "gpmodel"

    example["t"] = jnp.linspace(-0.8, 0.8, 1000)

    example["transit"] = example_transit_model(example["t"], [0.1, 0.1, 0.3])

    example["background"] = 0.002 * (
        5 * example["t"] ** 2
        + jnp.sin(20 * example["t"])
        + 0.3 * jnp.cos(50 * example["t"])
    )

    white_noise = noise_std * jr.normal(noise_key, (len(example["t"]),))
    noise = white_noise
    noise = jnp.zeros(len(example["t"]))
    noise = noise.at[0].set(white_noise[0])
    for i in range(1, len(example["t"])):  # generate ar(1) noise
        noise = noise.at[i].set(example["phi"] * noise[i - 1] + white_noise[i])
    example["noise"] = noise
    example["noise_std"] = noise_std / jnp.sqrt(1 - example["phi"] ** 2)

    train_ind = jnp.sort(
        jr.choice(idx_key, len(example["t"]), (num_train,), replace=False)
    )

    example["t_sample"] = example["t"][train_ind]
    example["lc_sample"] = (
        example["transit"] + example["background"] + example["noise"]
    )[train_ind]

    example["transit_mask"] = jnp.isclose(example["transit"], 0)
    example["transit_mask_sample"] = example["transit_mask"][train_ind]
    return example


def allan_deviation(data: Array | NDArray | list) -> tuple[NDArray, NDArray]:
    """Calculate Allan devation for a sample of
    time series.

    Parameters
    ----------
    data : Array | NDArray | list
        The data, must be of shape (n_samples, n_data).

    Returns
    -------
    tuple[NDArray | NDArray]
    The bin sizes, and Allan deviations (of
    shape (n_samples, n_data//2) ).
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_samples, n_data = data.shape

    def calculate_bin_size(bin_size: int, values: NDArray) -> NDArray:
        n_bins = n_data // bin_size
        reshaped_arr = data[:, : n_bins * bin_size].reshape(n_samples, n_bins, bin_size)

        # Calculate mean for each bin
        mean = np.mean(reshaped_arr, axis=2)

        # calculate variance for each sample
        variance = np.var(mean, axis=1, ddof=1)

        # Update values for the current bin size
        values[:, bin_size - 1] = variance

        return values

    # Initialize the array for storing Allan deviation values
    allan_deviation_values = np.zeros((n_samples, n_data // 2))

    # Compute Allan deviation for all bin sizes using a for loop
    bin_sizes = np.arange(1, n_data // 2 + 1)
    for bin_size in bin_sizes:
        allan_deviation_values = calculate_bin_size(bin_size, allan_deviation_values)

    return bin_sizes, allan_deviation_values


def allan_deviation_chi2_regions(bin_sizes: NDArray | Array, n_data: int) -> NDArray:
    """Calculate the expected mean and percentiles for the binned residuals in the
    Allan Deviation, for whitened resiudals.

    The whitened residuals are assumed to follow a standard normal distribution
    (mu = 0, var = 1). Binning the data down, where each bin contains the mean of
    n datapoints, the variance reduces as 1/N.

    However, for a sample of n_data datapoints, the variance estimate will itself be
    uncertain and follows a scaled chi2 distribution. This function calculates the
    mean and percentiles of the chi2 for every binning used in the Allan Deviation.

    Parameters
    ----------
    bin_sizes : NDArray | Array
        A list with the bin_sizes considered
    n_data : int
        The total number of datapoints for the residuals.

    Returns
    -------
    NDArray
        The chi square statistics for all bin sized, of shape (5, bin_sizes),
        where the rows are [mean, 16th percentile, 84th percentile,
        2.5th percentile, 97.5th percentile].
    """

    values = []

    for bin_size in bin_sizes:
        n_data_binned = n_data // bin_size
        degrees_of_freedom = n_data_binned - 1
        variance = 1 / bin_size

        scale_factor = variance / degrees_of_freedom

        mean = chi2.mean(degrees_of_freedom) * scale_factor
        percentiles = (
            np.array(
                [chi2.ppf(x, degrees_of_freedom) for x in [0.16, 0.84, 0.025, 0.975]]
            )
            * scale_factor
        )

        values.append(np.concatenate([[mean], percentiles]))

    return np.array(values).T
