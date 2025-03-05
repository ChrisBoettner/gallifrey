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

from gallifrey.model import GPConfig, GPModel
from gallifrey.schedule import LinearSchedule

# SETTING UP PATHS/SETTINGS
# get get path to save directory
save_path = (
    pathlib.Path(__file__).parent.parent / "model_checkpoints" / "quickstart_example"
)
if not save_path.exists():
    save_path.mkdir(parents=True)

# %%
# MOCK DATA
rng_key = jr.PRNGKey(1521)

key, data_key = jr.split(rng_key)
n = 220
noise_var = 9.0
x = jnp.linspace(0, 20, n)
y = (x + 0.01) * jnp.sin(x * 3.2) + jnp.sqrt(noise_var) * jr.normal(data_key, (n,))


# mask values
xtrain = x[(x < 10)]
ytrain = y[(x < 10)]


# %%
# SET UP GP MODEL

config = GPConfig()

key, model_key = jr.split(key)
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
    annealing_schedule=LinearSchedule().generate(len(xtrain), len(xtrain)),
    n_mcmc=75,
    n_hmc=10,
    verbosity=0,
)

gpmodel = gpmodel.update_state(final_smc_state)

# %%
# SAVE MODEL

gpmodel.save_state(str(save_path / "final_state.pkl"), final_smc_state)
gpmodel.save_state(str(save_path / "history.pkl"), history)
