import os

from beartype import BeartypeConf
from beartype.claw import beartype_this_package
from jax import config as jax_config

# type checks the entire package, but only throw warnings
# (must be before gallifrey imports)
beartype_this_package(conf=BeartypeConf(violation_type=UserWarning))  # type: ignore

from gallifrey.model import GPConfig, GPModel  # noqa: E402
from gallifrey.schedule import LinearSchedule, LogSchedule  # noqa: E402

os.environ["JAX_ENABLE_X64"] = "True"
jax_config.update("jax_enable_x64", True)
# no idea why, but this increases performance by a factor of 2, at least on CPU
os.environ["OMP_NUM_THREADS"] = "1"

print("gallifrey: Setting flag `JAX_ENABLE_X64` to `True`")
print("gallifrey: Setting flag `OMP_NUM_THREADS` to `1`")


__all__ = [
    "GPConfig",
    "GPModel",
    "LinearSchedule",
    "LogSchedule",
]
