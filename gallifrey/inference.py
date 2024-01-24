import gpjax as gpx
from jax.scipy.linalg import cholesky, inv
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from beartype.typing import Callable, Optional
from gpjax.objectives import ConjugateMLL
from gpjax.typing import Array, ScalarFloat
from jax import jit
from jax.tree_util import tree_leaves

from .kernelsearch import jit_set_trainables
from .lightcurve import Transit


def log_likelihood_function(
    gp_posterior: gpx.gps.AbstractPosterior,
    lc_model: Callable,
    X: Array,
    y: Array,
    mask: Array,
    fix_gp: bool = False,
    negative: bool = False,
    compile: bool = True,
) -> Callable:
    """Create objective function for lightcurve fit.
    The probability is calculated in the following way:

    - Calculate marginal probability for GP outside of transit
    (background, using mask), this corresponds to the likelihood
    of the GP hyperparameter
    - Remove model lightcurve
    - Calculate probability of light curve data after
    removing the transit (inverse mask), with the GP conditioned
    on the background data

    The returned Callable calculates the log probability
    for an input dict 'params', which must be a dictonary
    with the keys 'gp_parameter' and 'lc_parameter', and
    the values being arrays of the parameter.

    Parameters
    ----------
    gp_posterior : gpx.gps.AbstractPosterior
        The gaussian process to use for
        background.
    lc_model : Callable
        Model that calculates the light curve.
    X : Array
        X data, usually time.
    y : Array
        y data, usually the light curve/flux.
    mask : Array
        Boolean mask that covers the transits.
    fix_gp : bool, optional
        Whether to fix the parameter of the GP to initial
        values. In that case, the 'gp_parameter' key in
        the params input has no effect, by default False
    negative : bool, optional
        If True, return negative of probability. By
        default False
    compile : bool, optional
        Whether to compile the objective using jit, by
        default True
    Returns
    -------
    Callable
        The callable objective function. Takes 'params' as
        input, which must be a dict.
    """
    constant = jnp.array(-1.0) if negative else jnp.array(1.0)

    # if not jnp.isclose(jnp.mean(y), 0):
    #     raise ValueError(
    #         "Data must be centered at 0. Please center data first using y - mean(y)."
    #     )

    D_background = gpx.Dataset(
        X=X[mask].reshape(-1, 1),
        y=y[mask].reshape(-1, 1),
    )
    D_transit = gpx.Dataset(
        X=X[~mask].reshape(-1, 1),
        y=y[~mask].reshape(-1, 1),
    )

    marginal_log_likelihood = ConjugateMLL()

    # fix gp variables to to initial values
    if fix_gp:
        # marginal log likelihood for background,
        # constitutes likelihood function for hyperparameter
        # conditioned on data outside of transit
        background_log_prob = marginal_log_likelihood(
            gp_posterior,
            D_background,
        )

        transit_dist = predictive_distribution(
            gp_posterior,
            D_transit.X,  # type: ignore
            dataset=D_background,
        )

        def objective(params: Array) -> ScalarFloat:
            # calculate lightcurve model
            lightcurve = lc_model(D_transit.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = (D_transit.y - lightcurve).reshape(-1)
            transit_log_prob = transit_dist.log_prob(res.reshape(-1))
            # return (negative of, if wanted) log probability
            return constant * jnp.nan_to_num(
                transit_log_prob + background_log_prob, nan=-jnp.inf
            )

    # adapt gp parameter at every step
    else:
        # indices of trainables for GP
        trainable_idx = jnp.argwhere(
            jnp.array(tree_leaves(gp_posterior.trainables()))
        ).reshape(-1)

        def objective(params: Array) -> ScalarFloat:
            # update the parameter of the posterior object
            updated_posterior = jit_set_trainables(
                gp_posterior.unconstrain(),
                jnp.array(params["gp_parameter"]),
                trainable_idx,
            ).constrain()

            # marginal log likelihood for background,
            # constitutes likelihood function for hyperparameter
            # conditioned on data outside of transit
            background_log_prob = marginal_log_likelihood(
                updated_posterior,
                D_background,
            )

            transit_dist = predictive_distribution(
                updated_posterior,
                D_transit.X,  # type: ignore
                dataset=D_background,
            )
            # calculate lightcurve model
            lightcurve = lc_model(D_transit.X, params["lc_parameter"])

            # remove lightcurve from observations and
            # calculate probability under GP model
            res = D_transit.y - lightcurve
            transit_log_prob = transit_dist.log_prob(res.reshape(-1))

            # return (negative of, if wanted) log probability
            return constant * jnp.nan_to_num(
                transit_log_prob + background_log_prob, nan=-jnp.inf
            )

    if compile:
        return jit(objective)
    return objective


def predictive_distribution(
    gp_posterior: gpx.gps.AbstractPosterior,
    input: Array,
    *,
    dataset: Optional[gpx.Dataset] = None,
    x: Optional[Array] = None,
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
    x : Array, optional
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
        if x is None or y is None:
            raise ValueError("Either dataset or x and y must be provided")

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Create dataset using x and y
        dataset = gpx.Dataset(x, y)

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


@jit
def calculate_whitened_residuals(
    y: Array,
    distribution: tfp.distributions.Distribution,
) -> Array:
    """Calculate whitened residuals for between
    data y and the predictive distribution through
    Cholesky decomposition.

    Parameters
    ----------
    y : Array
        y data, usually the light curve/flux.
    distribution : tfp.distributions.Distribution
        The predicitve distribution at the points
        x corresponding to y.

    Returns
    -------
    Array
        The whitened residuals.
    """
    residuals = y - distribution.mean()
    covariance_matrix = distribution.covariance()
    cholesky_matrix = cholesky(covariance_matrix)
    inverse_matrix = inv(cholesky_matrix)
    whitened_residuals = jnp.dot(inverse_matrix, residuals)
    return whitened_residuals
