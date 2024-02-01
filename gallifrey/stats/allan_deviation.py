import numpy as np
from beartype.typing import Iterable
from jaxtyping import Array
from numpy.typing import NDArray
from scipy.stats import chi2


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


def chi2_regions(
    n_data: int,
    bin_sizes: Iterable | int = 1,
    percentiles: Iterable | float = [16, 84, 2.5, 97.5],
    variance: float = 1,
) -> NDArray:
    """
    Calculate the expected mean and percentiles for the binned residuals in the
    Allan Deviation, for whitened resiudals.

    The whitened residuals are assumed to follow a standard normal distribution
    (mu = 0, var = 1). Binning the data down, where each bin contains the mean of
    n datapoints, the variance reduces as 1/N.

    However, for a sample of n_data datapoints, the variance estimate will itself be
    uncertain and follows a scaled chi2 distribution. This function calculates the
    mean and percentiles of the chi2 for every binning used in the Allan Deviation.

    Parameters
    ----------
    n_data : int
        The total number of datapoints for the residuals.
    bin_sizes : Iterable | float
        A list with the bin_sizes considered.
    percentiles : Iterable | float
        The percentiles to calculate, by default [0.16, 0.84, 0.025, 0.975].
    variance : float
        The initial variance for bin_size = 1. By default 1, which is the
        expected value for whitened resiudals.

    Returns
    -------
    NDArray
        The chi square statistics for all bin sized, of shape (5, bin_sizes),
        where the rows are [mean, 16th percentile, 84th percentile,
        2.5th percentile, 97.5th percentile], or specified bins.
    """

    values = []

    bin_sizes = np.atleast_1d(np.asarray(bin_sizes))
    percentiles = np.atleast_1d(np.asarray(percentiles))
    assert isinstance(bin_sizes, Iterable)  # type: ignore
    assert isinstance(percentiles, np.ndarray)  # type: ignore

    for bin_size in bin_sizes:
        n_data_binned = n_data // bin_size
        degrees_of_freedom = n_data_binned - 1
        binned_variance = variance / bin_size

        scale_factor = binned_variance / degrees_of_freedom

        mean = chi2.mean(degrees_of_freedom) * scale_factor
        regions = (
            np.array([chi2.ppf(x, degrees_of_freedom) for x in (percentiles / 100)])
            * scale_factor
        )

        values.append(np.concatenate([[mean], regions]))

    return np.array(values).T
