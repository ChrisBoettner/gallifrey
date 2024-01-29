import numpy as np
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
