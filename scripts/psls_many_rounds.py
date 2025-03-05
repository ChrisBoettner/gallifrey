# DESCRIPTION: Running gallifrey on Plato solar-like lightcurve, 100 rounds at max_depth 2, half data

# %%
# IMPORTS
import os
import pathlib

num_cores = 64
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"

import jax.numpy as jnp
import lightkurve as lk
import pandas as pd
from jax import random as jr

from gallifrey.kernels.atoms import Matern32Atom, ProductOperator, RBFAtom, SumOperator
from gallifrey.model import GPConfig, GPModel
from gallifrey.schedule import LinearSchedule

# SETTING UP PATHS/SETTINGS
# get get path to save directory
save_path = (
    pathlib.Path(__file__).parent.parent / "model_checkpoints" / "psls_many_rounds"
)
if not save_path.exists():
    save_path.mkdir(parents=True)

# %%
# GET AND PROCESS DATA

rng_key = jr.PRNGKey(1505)

full_lightcurve = pd.read_csv("../data/PSLS/0012069449.csv")
mask = (full_lightcurve["time [d]"] > 49.425) & (full_lightcurve["time [d]"] < 49.58)
sub_selection = full_lightcurve[mask]
transit_mask = (full_lightcurve["time [d]"] > 49.49) & (
    full_lightcurve["time [d]"] < 49.51
)

training_set = full_lightcurve[mask & ~transit_mask][::2]

xtrain = jnp.array(training_set["time [d]"])
ytrain = jnp.array(training_set["flux [ppm]"])


# %%
# SET UP GP MODEL

config = GPConfig(
    max_depth=2,
    atoms=[RBFAtom(), Matern32Atom()],
    operators=[SumOperator(), ProductOperator()],
    node_probabilities=jnp.array([1.0, 1.0, 0.5, 0.5]),
)

key, model_key = jr.split(rng_key)
gpmodel = GPModel(
    model_key,
    x=xtrain,
    y=ytrain,
    num_particles=num_cores,
    config=config,
)
# %%
# RUN SMC

key, smc_key = jr.split(key)
final_smc_state, history = gpmodel.fit_smc(
    smc_key,
    annealing_schedule=LinearSchedule().generate(len(xtrain), 100),
    n_mcmc=75,
    n_hmc=10,
    verbosity=0,
)

gpmodel = gpmodel.update_state(final_smc_state)

# %%
# SAVE MODEL

gpmodel.save_state(str(save_path / "final_state.pkl"), final_smc_state)
gpmodel.save_state(str(save_path / "history.pkl"), history)
