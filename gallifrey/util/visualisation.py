import jax.numpy as jnp
import matplotlib.pyplot as plt
from beartype.typing import Any, Optional
from jaxtyping import Array
from tensorflow_probability.substrates.jax.distributions import Distribution

from gallifrey.util.allan_deviation import allan_deviation, allan_deviation_chi2_regions

colors = [
    "#2F83B4",
    "#B5121B",
    "#F77F00",
    "#0B6E4F",
    "#7A68A6",
    "#C5BB36",
    "#8c564b",
    "#e377c2",
]


def plot_prediction(
    ax: plt.Axes,  # type: ignore
    t: Array,
    distribution: Distribution | Array,
    errorbar: int = 0,
    kws_mean: Optional[dict] = None,
    kws_error: Optional[dict] = None,
    outline: bool = True,
) -> plt.Axes:  # type: ignore
    """Plot mean and standard deviation
    of a distribution.
    Input distribution can either be a distribution object,
    in which case mean and standard devation are plotted,
    or a sample from the distribution. In the second case,
    median and percentile region are returned.

    Parameters
    ----------
    ax : plt.Axes
        The figure axis.
    t : Array
        The points at which to evaluate
        the distribution, must match
        dimension of the distribtion.
    distribution : Distribution | Array
        The predictive distribution to
        evaluate, either a distribution instance
        or a sample from the distribution
        with shape (n_samples, len(t)).
    errorbar : int, optional
        In case of distribution instance, this
        corresponds to the number of std deviations
        to plot. In case of a sample, this corresponds
        to the percentiles that should be contained within
        the shaded region, by default 0
    kws_mean : Optional[dict], optional
        Additional arguments for the mean line,
        by default None
    kws_error : Optional[dict], optional
        Additional arguments for the shaded
        credible regions, by default None
    outline : bool, optional
        If True, credible regions are outlined,
        by default True

    Returns
    -------
    plt.Axes
        The figure axis.
    """
    kws_mean = {} if kws_mean is None else kws_mean
    kws_error = {} if kws_error is None else kws_error

    if isinstance(distribution, Distribution):
        mean = distribution.mean()
        std = distribution.stddev()
        values = jnp.array(
            [mean, mean - errorbar * std, mean + errorbar * std],  # type: ignore
        )
        kws_mean["label"] = kws_mean.get("label", "Mean")
        kws_error["label"] = kws_error.get("label", f"{int(errorbar)} Sigma")
    else:
        edges = (100 - errorbar) / 2
        values = jnp.percentile(
            distribution,
            jnp.array([50, edges, 100 - edges]),
            axis=0,
        )
        kws_mean["label"] = kws_mean.get("label", "Median")
        kws_error["label"] = kws_error.get(
            "label", rf"{int(errorbar)}\% Credible Region"
        )

    kws_mean["color"] = kws_mean.get("color", colors[1])
    kws_error["linewidth"] = kws_error.get("linewidth", 1)
    kws_error["linestyle"] = kws_error.get("linestyle", "--")
    kws_error["color"] = kws_error.get("color", kws_mean["color"])
    kws_error["alpha"] = kws_error.get("alpha", 0.2)

    ax.plot(
        t,
        values[0],
        **kws_mean,
    )

    if errorbar > 0:
        ax.fill_between(
            t,
            values[1],  # type: ignore
            values[2],  # type: ignore
            **kws_error,
        )
        if outline:
            for val in values[1:]:  # type: ignore
                ax.plot(
                    t,
                    val,
                    linestyle=kws_error["linestyle"],
                    linewidth=kws_error["linewidth"],
                    color=kws_error["color"],
                )
    return ax


def plot_masks(
    ax: plt.Axes,  # type: ignore
    t: Array,
    mask: Array,
    **kwargs: Any,
) -> plt.Axes:  # type: ignore
    """Shade masked regions.

    Parameters
    ----------
    ax : plt.Axes
        The figure axis.
    mask : Array
        The region to mask.

    Returns
    -------
    plt.Axes
        The figure axis.
    """
    ymin, ymax = ax.get_ylim()

    kwargs["alpha"] = kwargs.get("alpha", 0.3)
    kwargs["color"] = kwargs.get("color", "grey")
    kwargs["zorder"] = kwargs.get("zorder", 0)

    ax.fill_between(
        t,
        ymin,
        ymax,
        where=~mask,  # type: ignore
        **kwargs,
    )
    ax.set_ylim(ymin, ymax)
    return ax


def plot_residuals(
    ax: plt.Axes,  # type: ignore
    t: Array,
    whitened_residual_sample: Array,
    credible_region: int = 0,
    kws_residuals: Optional[dict] = None,
    kws_error: Optional[dict] = None,
) -> plt.Axes:  # type: ignore
    """Plot whitened residuals and
    the expected region.

    Parameters
    ----------
    ax : plt.Axes
        The figure axis.
    t : Array
        The points at which the resiudals are
        evaluated.
    whitened_residual_sample : Array
        An array of whitened residual samples,
        must be of shape (n_samples, len(t))
    credible_region : int, optional
        The size of the errorbars for the residuals,
        credible region contained within errorbars, by
        default 0
    kws_residuals : Optional[dict], optional
        Additional arguemnts for the mean line,
        by default None
    kws_error : Optional[dict], optional
        Additional arguments for the shaded
        credible regions, by default None

    Returns
    -------
    plt.Axes
        The figure axis.
    """
    kws_residuals = {} if kws_residuals is None else kws_residuals
    kws_error = {} if kws_error is None else kws_error

    kws_residuals["fmt"] = kws_residuals.get("fmt", ".")
    kws_residuals["label"] = kws_residuals.get(
        "label", rf"Median Residuals with {credible_region}\% credible region"
    )
    kws_error["color"] = kws_error.get("color", colors[1])
    kws_error["alpha"] = kws_error.get("alpha", 0.3)
    kws_error["alpha"] = kws_error.get("alpha", 0.3)

    if whitened_residual_sample.ndim == 1:
        whitened_residual_sample = whitened_residual_sample.reshape(1, -1)

    edges = (100 - credible_region) / 2
    percentiles = jnp.percentile(
        whitened_residual_sample,
        jnp.array([50, edges, 100 - edges]),
        axis=0,
    )

    ax.errorbar(
        t,
        percentiles[0],
        [
            percentiles[0] - percentiles[1],
            percentiles[2] - percentiles[0],
        ],
        **kws_residuals,
    )

    t_diff = jnp.abs(t.max() - t.min())
    t_lims = [
        t.min() - 0.05 * t_diff,
        t.max() + 0.05 * t_diff,
    ]

    ax.axhline()
    for r, percentage, alpha_adj in [([-2, 2], 95, 0), ([-1, 1], 68, 0.2)]:
        ax.fill_between(
            t_lims,
            r[0],
            r[1],
            alpha=kws_error["alpha"] + alpha_adj,
            # label=rf"Expected {percentage}\% credible region",
            **{k: v for k, v in kws_error.items() if k != "alpha"},
        )

    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlim(*t_lims)  # type: ignore
    return ax


def plot_allan_deviation(
    ax: plt.Axes,  # type: ignore
    whitened_residual_sample: Array,
    credible_region: int = 0,
    kws_residuals: Optional[dict] = None,
    kws_error: Optional[dict] = None,
) -> plt.Axes:  # type: ignore
    """Plot whitened residuals and
    the expected region.

    Parameters
    ----------
    ax : plt.Axes
        The figure axis.
    whitened_residual_sample : Array
        An array of whitened residual samples,
        must be of shape (n_samples, len(t))
    credible_region : int, optional
        The size of the errorbars for the residuals,
        credible region contained within errorbars, by
        default 0
    kws_residuals : Optional[dict], optional
        Additional arguemnts for the mean line,
        by default None
    kws_error : Optional[dict], optional
        Additional arguments for the shaded
        credible regions, by default None

    Returns
    -------
    plt.Axes
        The figure axis.
    """
    kws_residuals = {} if kws_residuals is None else kws_residuals
    kws_error = {} if kws_error is None else kws_error

    kws_residuals["fmt"] = kws_residuals.get("fmt", ".")
    kws_residuals["label"] = kws_residuals.get(
        "label", rf"Median Residuals with {credible_region}\% credible region"
    )
    kws_error["color"] = kws_error.get("color", colors[1])
    kws_error["alpha"] = kws_error.get("alpha", 0.3)
    kws_error["alpha"] = kws_error.get("alpha", 0.3)

    if whitened_residual_sample.ndim == 1:
        whitened_residual_sample = whitened_residual_sample.reshape(1, -1)

    # calculate allan deviation
    bin_sizes, allan_deviation_sample = allan_deviation(
        whitened_residual_sample
    )  # type : ignore

    expected_deviation = allan_deviation_chi2_regions(
        bin_sizes, whitened_residual_sample.shape[-1]
    )

    # calculate percentiles
    edges = (100 - credible_region) / 2
    percentiles = jnp.percentile(
        allan_deviation_sample,
        jnp.array([50, edges, 100 - edges]),
        axis=0,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.errorbar(
        bin_sizes,
        percentiles[0],
        [
            percentiles[0] - percentiles[1],
            percentiles[2] - percentiles[0],
        ],
        **kws_residuals,
    )

    ax.plot(
        bin_sizes,
        expected_deviation[0],
        color=kws_error["color"],
        # label=r"$1/N$",
    )

    expected_68 = [expected_deviation[1], expected_deviation[2]]
    expected_95 = [expected_deviation[3], expected_deviation[4]]

    for r, percentage, alpha_adj in [(expected_95, 95, 0), (expected_68, 68, 0.2)]:
        ax.fill_between(
            bin_sizes,
            r[0],
            r[1],
            alpha=kws_error["alpha"] + alpha_adj,
            # label=rf"Expected {percentage}\% credible region",
            **{k: v for k, v in kws_error.items() if k != "alpha"},
        )
    return ax
