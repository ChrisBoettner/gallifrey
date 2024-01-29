import gpjax as gpx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from beartype.typing import Callable, Optional
from gpjax.typing import Array
from jax.tree_util import tree_leaves

from gallifrey.gps.trainables import jit_set_trainables
from gallifrey.transit.gpjax import Transit


def predictive_distribution(
    gp_posterior: gpx.gps.AbstractPosterior,
    input: Array,
    *,
    dataset: Optional[gpx.Dataset] = None,
    X: Optional[Array] = None,
    y: Optional[Array] = None,
    transit_model: Optional[Callable] = None,
    transit_parameter: Optional[Array] = None,
    gp_parameter: Optional[Array] = None,
) -> tfp.distributions.Distribution:
    """
    Calculate the predictive distribution of
    the GP for a given input (x data),
    under the observations, which can be either given as
    a gpjax Dataset or using x, y from which the Dataset
    gets constructed.
    Optionally, a transit model with transit parameter
    can be given to include the transit as a mean function.

    Parameters
    ----------
    gp_posterior : gpx.gps.AbstractPosterior
        The GPJax posterior object.
    input : Array
        The x values at which to calculate the distribution.
    dataset : gpx.Dataset, optional
        The training data to condition the GP on.
    X : Array, optional
        The x values of the training data.
    y : Array, optional
        The y values of the training data.
    transit_model : Optional[Callable], optional
        The transit model, must be a function of the form
        f(x, parameter) -> y, that takes the points at which
        to evaluate the function and parameter as input, and
        returns the flux at the input points, by default
        None
    transit_parameter : Optional[Array], optional
        The parameter for the transit model. Must be given
        if the transit model is given, by default None
    gp_parameter : Optional[Array], optional
        If given, updates the gp parameter to these values before
        calculating the distribution, by default None

    Returns
    -------
    tfp.distributions.Distribution
        The predictive distribution. (Most likely
        multivariate Gaussian)
    """

    # Reshape input if needed
    if input.ndim == 1:
        input = jnp.asarray(input).reshape(-1, 1)

    # Check if dataset is not provided
    if dataset is None:
        if X is None or y is None:
            raise ValueError("Either dataset or x and y must be provided")

        X = jnp.asarray(X)
        y = jnp.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Create dataset using x and y
        dataset = gpx.Dataset(X, y)

    if gp_parameter is not None:
        # update gp parameter
        trainable_idx = jnp.argwhere(
            jnp.array(tree_leaves(gp_posterior.trainables()))
        ).reshape(-1)
        gp_posterior = jit_set_trainables(
            gp_posterior.unconstrain(),
            gp_parameter,  # type: ignore
            trainable_idx,
        ).constrain()

    if transit_model is not None:
        # add transit model as mean function
        if transit_parameter is None:
            raise ValueError(
                "If transit_model is given, transit_parameter must be as well."
            )
        transit = Transit(transit_model, transit_parameter)
        gp_prior = gpx.gps.Prior(
            mean_function=transit,
            kernel=gp_posterior.prior.kernel,
        )
        gp_posterior = gp_posterior.likelihood * gp_prior

    latent_dist = gp_posterior(input, train_data=dataset)
    predictive_dist = gp_posterior.likelihood(
        latent_dist
    )  # adds observational uncertainty
    return predictive_dist
