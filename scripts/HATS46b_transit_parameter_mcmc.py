# DESCRIPTION: Running MCMC on transit parameters for HATS46b, using saved GP models

# %%
# IMPORTS
import os

num_cores = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"

# import libraries
import pathlib
import pickle
from collections import OrderedDict

import blackjax
import jax
from astropy.table import Table
from blackjax.util import run_inference_algorithm
from jax import numpy as jnp
from jax import random as jr
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.keplerian import Body, Central, System

from gallifrey.kernels.atoms import (
    LinearAtom,
    Matern12Atom,
    Matern32Atom,
    Matern52Atom,
    ProductOperator,
    RBFAtom,
    SumOperator,
)
from gallifrey.model import GPConfig, GPModel

# %%
# SET UP PATH AND RNG KEY

path = pathlib.Path(__file__).parent.parent

# set a random key for for this notebook
rng_key = jr.PRNGKey(7)

# %%
# LOAD DATA
data = (
    Table.read(path / "data/HATS46b/HATS_46b.fits")
    .to_pandas()
    .drop(columns=["FWB20", "e_FWB20"])  # not used in paper
    .rename(columns={"Time": "Time [MJD]"})
)

time = jnp.array(data["Time [MJD]"].values)
time_zero = time[0]
time -= time_zero

# spectroscopic and white light curves, initial entry is white lc
flux = jnp.array(data.iloc[:, 1::2].values).T
e_flux = jnp.array(data.iloc[:, 2::2].values).T  # uncertainties

# mask out transit
transit_mask = jnp.where(
    (time > time[6]) & (time < time[41]),
    True,
    False,
)

PLANET_PERIOD = 4.7423749  # in days, reference from arXiv:2303.07381

num_datasets = len(flux)

# %%
# LOAD GP MODELS

config = GPConfig(
    max_depth=2,
    atoms=[LinearAtom(), RBFAtom(), Matern12Atom(), Matern32Atom(), Matern52Atom()],
    operators=[SumOperator(), ProductOperator()],
    node_probabilities=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
)

gpmodels = []
key = rng_key
for i in range(num_datasets):
    key, model_key = jr.split(key)
    gpmodel = GPModel(
        model_key,
        x=time[~transit_mask],
        y=flux[i, ~transit_mask],
        num_particles=8,
        config=config,
    )
    gpmodels.append(gpmodel)

model_names = data.columns[1::2]

for i in range(num_datasets):
    gpmodel = gpmodels[i]
    final_smc_state = gpmodel.load_state(
        str(path / f"model_checkpoints/HATS46b/final_state_{model_names[i]}.pkl")
    )
    gpmodels[i] = gpmodel.update_state(final_smc_state)

# %%
# CREATE TRANSIT MODEL


def transit_model(params, shared_params, time, period):
    # define planetary system model with shared and individual parameters
    central_body = Central(
        mass=shared_params["central_mass"],
        radius=shared_params["central_radius"],
    )

    orbiting_planet = Body(
        period=period,
        radius=params["radius_ratio"] * shared_params["central_radius"],
        inclination=shared_params["inclination"],
        time_transit=shared_params["t0"],
    )

    stellar_system = System(central_body, bodies=[orbiting_planet])

    # return the light curve
    return limb_dark_light_curve(
        stellar_system, jnp.array([params["u1"], params["u2"]])
    )(time).squeeze()


# vectorize the transit model for all datasets
transit_model = jax.vmap(transit_model, in_axes=(0, None, None, None))

# %%
# SETUP OBJECTIVE

from tensorflow_probability.substrates.jax.bijectors import Sigmoid, Softplus

bijectors = {
    "radius_ratio": Sigmoid(low=0.0, high=1.0),
    "u1": Sigmoid(low=0.3, high=1.0),
    "u2": Sigmoid(low=-0.1, high=0.2),
    "central_mass": Sigmoid(low=0.8, high=0.9),
    "central_radius": Sigmoid(low=0.85, high=0.95),
    "inclination": Sigmoid(low=jnp.deg2rad(80.0), high=jnp.deg2rad(90.0)),
    "t0": Sigmoid(low=0.05, high=0.09),
}


def transform_params(params, bijectors, direction="forward"):
    shared_params_dict = params["shared"]
    individual_params_dict = params["individual"]

    if direction == "forward":
        bijection_funcs = {key: bijectors[key].forward for key in bijectors.keys()}
    elif direction == "inverse":
        bijection_funcs = {key: bijectors[key].inverse for key in bijectors.keys()}

    shared_params = OrderedDict(
        {key: bijection_funcs[key](value) for key, value in shared_params_dict.items()}
    )
    individual_params = OrderedDict(
        {
            key: bijection_funcs[key](value)
            for key, value in individual_params_dict.items()
        }
    )
    return OrderedDict({"shared": shared_params, "individual": individual_params})


time_norm = gpmodels[0].x_transform(time)  # same for all models
time_transit = time_norm[transit_mask]

# create predictive distributions for each dataset (only for the transit region)
predictive_gmms = [
    gpmodel.get_mixture_distribution(time_transit) for gpmodel in gpmodels
]


def objective(params, time, flux, predictive_gmms, gpmodels, bijectors):

    # transform the parameters to the original space
    constrained_params = transform_params(params, bijectors, direction="forward")

    # calculate the transit light curves and residuals
    transit_light_curves = transit_model(
        constrained_params["individual"],
        constrained_params["shared"],
        time,
        PLANET_PERIOD,
    )

    residuals = flux - transit_light_curves

    # calculate the log probability for each light curve
    log_probs = jnp.array(
        [
            predictive_gmm.log_prob(gpmodel.y_transform(residual))
            for residual, gpmodel, predictive_gmm in zip(
                residuals, gpmodels, predictive_gmms
            )
        ]
    )
    # sum the log probabilities to get the total log probability
    log_prob = jnp.sum(log_probs)
    return log_prob


jitted_objective = jax.jit(
    lambda params: objective(
        params,
        time[transit_mask],
        flux[:, transit_mask],
        predictive_gmms,
        gpmodels,
        bijectors,
    )
)


initial_shared_params = OrderedDict(
    central_mass=jnp.array(0.869),
    central_radius=jnp.array(0.894),
    inclination=jnp.deg2rad(86.97),
    t0=jnp.array(0.07527),
)

initial_individual_params = OrderedDict(
    radius_ratio=jnp.full(num_datasets, 0.1125),
    u1=jnp.full(num_datasets, 0.547),
    u2=jnp.full(num_datasets, 0.1171),
)

initial_params = OrderedDict(
    shared=initial_shared_params,
    individual=initial_individual_params,
)

# transform parameters to unconstrained space
initial_params_uncontrained = transform_params(
    initial_params, bijectors, direction="inverse"
)

# %%
# RUN MCMC

# parameter adaption and burn-in
warmup = blackjax.window_adaptation(blackjax.nuts, jitted_objective, progress_bar=True)
key, warmup_key, sample_key = jax.random.split(key, 3)
(burned_in_state, nuts_parameters), _ = warmup.run(
    warmup_key, initial_params_uncontrained, num_steps=1000
)

# sampling
nuts_sampler = blackjax.nuts(jitted_objective, **nuts_parameters)

inference_algorithm = lambda rng_key: run_inference_algorithm(
    rng_key=rng_key,
    inference_algorithm=nuts_sampler,
    num_steps=int(3e3),
    initial_state=burned_in_state,
    progress_bar=True,
)

final_state, (history, info) = jax.pmap(inference_algorithm)(jr.split(sample_key, 8))


mcmc_chains = transform_params(history.position, bijectors, direction="forward")

with open(path / "data/interim/mcmc_chains/HATS46b.pkl", "wb") as file:
    pickle.dump(mcmc_chains, file)
