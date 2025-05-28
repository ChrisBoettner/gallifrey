# gallifrey
[![Documentation](https://img.shields.io/badge/docs-main-red.svg)](https://chrisboettner.github.io/gallifrey/)
[![PyPI version](https://badge.fury.io/py/gallifrey.svg)](https://pypi.org/project/gallifrey/)
[![DOI](https://zenodo.org/badge/DOI/10.1051/0004-6361/202554518.svg)](https://doi.org/10.1051/0004-6361/202554518)
[![License](https://img.shields.io/badge/License-Apache%202.0-white.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

gallifrey is a package for structure discovery in time series data using Gaussian Processes with explicit applications to exoplanet transit lightcurves. 
It is a JAX-based python implementation of the julia package [AutoGP.jl](https://probsys.github.io/AutoGP.jl/stable/index.html) by Feras Saad.

<video autoplay muted loop>
    <source src="./assets/transit_animation.webm">
</video>

## Installation

**Option 1:  Installation using pip (Recommended)**

```bash
pip install gallifrey
```

**Option 2: Installation from Source**

1.  Clone the repository:

    ```bash
    git clone git@github.com:ChrisBoettner/gallifrey.git
    cd gallifrey
    ```

2.  Install the package:

    ```bash
    pip install .
    ```
    (or, for development mode: `pip install -e .`)

## Citation

If you use gallifrey in your research, please cite it as:

```bibtex
@article{https://doi.org/10.1051/0004-6361/202554518,
  doi = {10.1051/0004-6361/202554518},
  author = {Boettner, Christopher},
  title = {gallifrey: JAX-based Gaussian Process Structure Learning for Astronomical Time Series},
  year = {2025},
  journal = {A\&A},
  publisher = {EDP Sciences},
  issn = {0004-6361, 1432-0746},
  eprint = {2505.20394},
  archiveprefix = {arXiv},
  primaryclass = {astro-ph},
  keywords = {Astrophysics - Earth and Planetary Astrophysics,Astrophysics - Instrumentation and Methods for Astrophysics},
  copyright = {{\copyright} 2025, ESO},
}
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