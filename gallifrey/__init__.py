from jax import config
import logging

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)

# Throw error if any of the jax operations evaluate to nan.
#config.update("jax_debug_nans", True)

# silence jaxoplanet unit override
logging.getLogger("pint").setLevel(logging.ERROR)
