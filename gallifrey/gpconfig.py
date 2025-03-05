from dataclasses import field

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from flax import struct
from jaxtyping import Float
from tensorflow_probability.substrates.jax.distributions import (
    Distribution,
    InverseGamma,
)

from gallifrey.kernels.atoms import (  # noqa: F401
    AbstractAtom,
    AbstractOperator,
    ConstantAtom,
    LinearAtom,
    LinearWithShiftAtom,
    Matern12Atom,
    Matern32Atom,
    Matern52Atom,
    PeriodicAtom,
    PoweredExponentialAtom,
    ProductOperator,
    RationalQuadraticAtom,
    RBFAtom,
    SumOperator,
    WhiteAtom,
)


@struct.dataclass
class GPConfig:
    """
    Config for the GP model.

    Attributes
    ----------

    max_depth : int
        Maximum depth of the kernel tree. By default, 3.

    atoms : list[AbstractAtom]
        List of atomic kernels to consider in the kernel tree.
        By default, the following kernels are included:
        - Linear
        - Periodic
        - RBF

    operators : list[AbstractOperator]
        List of kernel operators to consider in the kernel tree.
        The operators are used to combine the atomic kernels (i.e., functions)
        in the kernel structure. By default, the following operators are included:
        - SumOperator
        - ProductOperator

    node_probabilities :  Float[jnp.ndarray, " D"]
        Probabilities for sampling the kernels and operators. This array
        should have a length equal to the sum of the number of kernels
        and operators. The first part of the array should contain the
        probabilities for sampling the kernels (in the order of them being listed
        in the `atoms` attribute), and the second part should contain the
        probabilities for sampling the operators (in the order of them being listed
        in the `operators` attribute).
        By default, the probabilities are set to be equal for all kernels, and
        half that probability for the operators (to encourage kernels with
        fewer terms)

    prior_transforms : dict[str, Transformation]
        Dictionary containing the bijectors for transforming the kernel parameters
        to the prior distribution. Originally, the parameters are sampled from a
        standard normal distribution, and these bijectors transform the samples to
        the desired prior distribution. The keys are inherited from GPJax, and describe
        the domain of the kernel parameters. The values are the bijectors that
        transform the standard normal samples to the desired prior distribution. The
        bijection transformation should be implemented via TensorFlow Probability
        bijectors.
        By default, the following bijectors are included:
        - "real": Log-normal transform (normal -> log-normal)
        - "positive": Log-normal transform (normal -> log-normal)
        - "sigmoid": Logit-normal transform (normal -> logit-normal)
        NOTE: The "none" key is reserved for internal use and should not be used.

    hmc_config : dict[str, float]
        Configuration for the HMC sampler. The dictionary should contain the following
        keys:
        - "step_size": The step size of the HMC sampler. By default, 0.02.
        - "inv_mass_matrix_scaling": The scaling factor for the inverse mass matrix.
            By default, 1.0.
        - "num_integration_steps": The number of integration steps for the HMC sampler.
            By default, 10.

    noise_prior : Distribution
        An instance of tensorflow_probability.substrates.jax.distributions.Distribution
        that samples the noise variance from the prior distribution. Must have methods
        "sample" and "log_prob". The default is an InverseGamma(1,1) distribution.
        NOTE: The noise variance prior is currently fixed to the InverseGamma(1,1)
        distribution, and cannot be changed. This is because the InverseGamma(1,1)
        distribution is needed to caluculate the Monte Carlo acceptance probabilities
        analytically. Other distributions are not currently supported.

    mean_function : AbstractMeanFunction
        The mean function of the GP model. By default, a zero mean function is used.
        The constant is explicitly set to 0.0, (and is not a ParticleParameter from
        the gallifrey.parameter class, so it is not trainable).
        NOTE: Non-zero mean functions are not currently implemented, do not change
        this attribute.

    """

    max_depth: int = 3

    atoms: list[AbstractAtom] = field(
        default_factory=lambda: [
            # ConstantAtom(),
            LinearAtom(),
            # LinearWithShiftAtom(),
            PeriodicAtom(),
            # Matern12Atom(),
            # Matern32Atom(),
            # Matern52Atom(),
            RBFAtom(),
            # PoweredExponentialAtom(),
            # RationalQuadraticAtom(),
            # WhiteAtom(),
        ]
    )

    operators: list[AbstractOperator] = field(
        default_factory=lambda: [
            SumOperator(),
            ProductOperator(),
        ]
    )

    node_probabilities: Float[jnp.ndarray, " D"] = field(
        default_factory=lambda: jnp.array(
            [
                # 1.0,  # Constant
                1.0,  # Linear
                # 1.0, # LinearWithShift
                1.0,  # Periodic
                # 1.0, # Matern12
                # 1.0, # Matern32
                # 1.0, # Matern52
                1.0,  # RBF
                # 1.0, # PoweredExponential
                # 1.0, # RationalQuadratic
                # 1.0, # White
                0.5,  # SumOperator
                0.5,  # ProductOperator
            ]
        )
    )

    prior_transforms: dict[str, tfb.Bijector] = field(
        default_factory=lambda: dict(
            {
                # this is the transformation y = exp(mu + sigma * z),
                # with mu = 0 and sigma = 1,
                # if z ~ normal(0, 1) then y ~ log-normal(mu, sigma)
                "real": tfb.Chain(
                    [
                        tfb.Exp(),
                        tfb.Shift(jnp.array(0.0)),
                        tfb.Scale(jnp.array(1.0)),
                    ]
                ),
                "positive": tfb.Chain(
                    [
                        tfb.Exp(),
                        tfb.Shift(jnp.array(0.0)),
                        tfb.Scale(jnp.array(1.0)),
                    ]
                ),
                # this is the transformation y = 1/(1 + exp(-(mu + sigma * z))),
                # with mu = 0 and sigma = 1,
                # if z ~ normal(0, 1) then y ~ logit-normal(mu, sigma)
                "sigmoid": tfb.Chain(
                    [
                        tfb.Sigmoid(
                            low=jnp.array(0.0),
                            high=jnp.array(
                                0.95  # slightly below 1 to avoid numerical issues
                            ),
                        ),
                        tfb.Shift(jnp.array(0.0)),
                        tfb.Scale(jnp.array(1.0)),
                    ]
                ),
            }
        )
    )

    hmc_config: dict[str, float] = field(
        default_factory=lambda: {
            "step_size": 0.02,
            "inv_mass_matrix_scaling": 1.0,
            "num_integration_steps": 10,
        }
    )

    @property
    def noise_prior(self) -> Distribution:
        """
        The distribution of the noise variance prior,
        specifically an InverseGamma(1,1) distribution.

        Returns
        -------
        Distribution
            The noise variance prior distribution.
        """
        return InverseGamma(jnp.array(1.0), jnp.array(1.0))
