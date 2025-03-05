# $\texttt{gallifrey}$: Bayesian Time Series Structure Learning with Gaussian Processes

[![Documentation](https://img.shields.io/badge/docs-main-red.svg)](https://chrisboettner.github.io/gallifrey/)
[![PyPI version](https://badge.fury.io/py/gallifrey.svg)](https://pypi.org/project/gallifrey/)
[![DOI](https://zenodo.org/badge/DOI/tba.svg)](https://doi.org/10.000)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

$\texttt{gallifrey}$ is a Python package designed for  Bayesian structure learning, inference, and analysis with Gaussian Process (GP) models, focused on time series data. It is a JAX-based python implementation of the julia package [AutoGP.jl](https://probsys.github.io/AutoGP.jl/stable/index.html) by Feras Saad. 

$\texttt{gallifrey}$ utilizes JAX for efficient numerical computation and Sequential Monte Carlo (SMC) methods for robust posterior approximation. Unlike most Gaussian Process packages, where a covariance function needs to be specified explicitly, $\texttt{gallifrey}$ infers the covariance structure from the time series.

 $\texttt{gallifrey}$ was created with exoplanet transit light curves in mind, but is applicable to a wide variety of time series modelling, analysis, and forecasting tasks.

![](./figures/animations/transit_animation.gif)

## Core Functionality
*   **Gaussian Process (GP) Modeling:**  Implements Gaussian Processes, leveraging JAX for efficient computation, with a particular focus on accurate uncertainty estimation.

*   **Bayesian Structure Learning:**  Provides a probabilistic framework for identifying latent structure within time series data by dynamically learning the covariance structure of the Gaussian Process.

*   **Sequential Monte Carlo (SMC):** Employs SMC for robust and fast posterior approximations.

## Installation

$\texttt{gallifrey}$ requires Python 3.10 or later.

**Option 1: Using pip (Recommended)**

```bash
pip install gallifrey
```

**Option 2: From source**

```bash
git clone git@github.com:ChrisBoettner/gallifrey.git
cd gallifrey
pip install .
```

For development (editable) installation:

```bash
pip install -e .
```

## Dependencies

$\texttt{gallifrey}$'s core functionality relies on the following packages:

*   `blackjax (>=1.2.5,<2.0.0)`
*   `jax (>=0.5.0,<0.6.0)`
*   `flax (>=0.10.3,<0.11.0)`
*   `equinox (>=0.11.11,<0.12.0)`
*   `beartype (>=0.19.0,<0.20.0)`
*   `tensorflow-probability (>=0.25.0,<0.26.0)`

## Quick Start

This example demonstrates a basic workflow, from data generation to model fitting and prediction.

```python
# Configure JAX to use all CPU cores
import multiprocessing
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

# Import necessary packages
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import seaborn as sns  # For plotting

# Import core components from gallifrey
from gallifrey.model import GPConfig, GPModel
from gallifrey.schedule import LinearSchedule

# Example Data Generation
rng_key = jr.PRNGKey(0)
key, data_key = jr.split(rng_key)
n = 120
noise_var = 9.0
x = jnp.linspace(0, 15, n)
y = (x + 0.01) * jnp.sin(x * 3.2) + jnp.sqrt(noise_var) * jr.normal(data_key, (n,))

# Split into training and test sets
xtrain = x[(x < 10)]
ytrain = y[(x < 10)]

# Model Initialization
config = GPConfig()  # Use default configuration (can be customized)
key, model_key = jr.split(key)
gpmodel = GPModel(
    model_key,
    x=xtrain,
    y=ytrain,
    num_particles=8,  # Number of particles for SMC
    config=config,
)

# Model Fitting (SMC)
key, smc_key = jr.split(key)
# Generate an annealing schedule (important for SMC)
annealing_schedule = LinearSchedule().generate(len(xtrain), 10)

final_smc_state, history = gpmodel.fit_smc(
    smc_key,
    annealing_schedule=annealing_schedule,
    n_mcmc=50,      # Number of MCMC steps per SMC iteration
    n_hmc=10,       # Number of HMC steps within each MCMC step
    verbosity=1,     # Control verbosity
)

# Update the model with the final SMC state
gpmodel = gpmodel.update_state(final_smc_state)

# Prediction
xtest = gpmodel.x_transform(jnp.linspace(0, 18, 60)) # Create x values for prediction
dist = gpmodel.get_mixture_distribution(xtest) # Get the predictive distribution

predictive_mean = dist.mean()
predictive_std = dist.stddev()

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(xtest, predictive_mean, label="Predictive Mean", color="C0")
plt.fill_between(
    xtest,
    predictive_mean - predictive_std,
    predictive_mean + predictive_std,
    alpha=0.3,
    label="Predictive Std. Dev.",
    color="C0"
)
plt.scatter(gpmodel.x_transformed, gpmodel.y_transformed, label="Training Data", color="C1", s=20)
plt.scatter(gpmodel.x_transform(x), gpmodel.y_transform(y), label="All Data", color="C2", s=10, alpha=0.5)
plt.show()
```

## Documentation and further examples

More detailed examples can be found in the `notebooks/` directory and the [documentation](https://chrisboettner.github.io/gallifrey/).

## Contributing

We welcome bug reports, feature requests, and pull requests.

## Citation

If you use $\texttt{gallifrey}$ in your research, please cite it as:

```bibtex
tba
```
And please also cite the original paper by Saad et al.

```bibtex
@article{https://doi.org/10.48550/arxiv.2307.09607,
  doi = {10.48550/ARXIV.2307.09607},
  url = {https://arxiv.org/abs/2307.09607},
  author = {Saad,  Feras A. and Patton,  Brian J. and Hoffman,  Matthew D. and Saurous,  Rif A. and Mansinghka,  Vikash K.},
  keywords = {Machine Learning (cs.LG),  Artificial Intelligence (cs.AI),  Methodology (stat.ME),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Sequential Monte Carlo Learning for Time Series Structure Discovery},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```

## Acknowledgements

This package is a direct re-implementation of [AutoGP.jl](https://probsys.github.io/AutoGP.jl/stable/index.html) and would not be possible without it. 
The Gaussian Procress implementation is strongly inspired by the fantastic packages [GPJax](https://docs.jaxgaussianprocesses.com/) and [tinygp](https://tinygp.readthedocs.io/en/stable/).