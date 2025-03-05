# %%
# IMPORTS
import os
import pathlib

num_cores = 64
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_cores}"


import jax.numpy as jnp
import lightkurve as lk
from jax import random as jr

from gallifrey.model import GPConfig, GPModel
from gallifrey.schedule import LinearSchedule

# SETTING UP PATHS/SETTINGS
# get get path to save directory
save_path = (
    pathlib.Path(__file__).parent.parent
    / "model_checkpoints"
    / "stellar_variability_extrapolation"
)
if not save_path.exists():
    save_path.mkdir(parents=True)

# %%
# GET AND PROCESS DATA

# get lightcurve
lcf = (
    lk.search_lightcurve("TIC 10863087", mission="TESS", author="SPOC", limit=1)
    .download(quality_bitmask="hard")
    .remove_nans()  # type: ignore
)
lc = lcf[100:5000:6].remove_outliers(sigma=3)

# convert to pandas dataframe
df = lc.to_pandas().reset_index().filter(["time", "flux", "flux_err"])
x = jnp.array(df["time"].values)
y = jnp.array(df["flux"].values)


xtrain = x[x < 1390]
ytrain = y[x < 1390]

# %%
# SET UP GP MODEL

config = GPConfig()

rng_key = jr.PRNGKey(12)

key, model_key = jr.split(rng_key)
gp_model = GPModel(
    model_key,
    x=xtrain,
    y=ytrain,
    num_particles=num_cores,
    config=config,
)

# %%
# RUN SMC

key, smc_key = jr.split(key)
final_smc_state, history = gp_model.fit_smc(
    smc_key,
    annealing_schedule=LinearSchedule().generate(len(xtrain), 20),
    n_mcmc=75,
    n_hmc=10,
    verbosity=1,
)
gp_model = gp_model.update_state(final_smc_state)

# %%
# SAVE MODEL

gp_model.save_state(str(save_path / "final_state.pkl"), final_smc_state)
gp_model.save_state(str(save_path / "history.pkl"), history)
