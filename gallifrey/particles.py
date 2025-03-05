#########################
# Code adapted from GPJax
#########################
from copy import deepcopy

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from flax import nnx
from jax import lax
from jax.scipy import linalg
from jaxtyping import Array, Float
from tensorflow_probability.substrates.jax.distributions import (
    Distribution,
    MultivariateNormalFullCovariance,
)

from gallifrey.data import Dataset
from gallifrey.kernels.prior import KernelPrior
from gallifrey.kernels.tree import TreeKernel
from gallifrey.parameter import NoiseParameter, transform_kernel_parameters
from gallifrey.utils.probability_calculations import calculate_covariance_matrix
from gallifrey.utils.typing import ScalarBool, ScalarFloat

SOFTPLUS_BIJECTOR = tfb.Softplus()


class Particle(nnx.Module):
    """
    The `Particle` class. This class is based on of the GPJax
    `ConjugatePosterior`.

    It takes a kernel (instance of `TreeKernel`), noise variance
    (float), and jitter (float) as input. The jitter is used to
    ensure that the covariance matrix is positive definite (avoiding
    numerical instabilities in case of small eigenvalues).

    In contrast to the `ConjugatePosterior` class, the `Particle` currently
    has more limited features, specifically we
        - do not support mean functions (that implies a zero mean function,
          in use cases the data should be appropriately centered).
        - do not support likelihoods other than Gaussian.

    Attributes
    ----------
    kernel : TreeKernel
        The kernel defining the covariance function of the Gaussian
        process. Must be a `TreeKernel` instance (from `gallifrey.kernels`).

    noise_variance : nnx.Variable | NoiseParameter
        The (observational) noise variance of the Gaussian process. Depending
        on the value of the `trainable_noise_variance` parameter during initialization,
        the noise variance will be either considered a trainable parameter (instance
        of `NoiseParameter`) or a fixed parameter (instance of `nnx.Variable`).

    jitter : ScalarFloat
        The jitter term to ensure numerical stability of the covariance
        matrix.
    """

    def __init__(
        self,
        kernel: TreeKernel,
        noise_variance: ScalarFloat,
        trainable_noise_variance: bool = False,
        jitter: float = 1e-6,
    ):
        """
        Initialize the particle.

        Parameters
        ----------
        kernel : TreeKernel
            The kernel defining the covariance function of the Gaussian
            process. Must be a `TreeKernel` instance (from `gallifrey.kernels`).
        noise_variance : ScalarFloat
            The (observational) noise variance of the Gaussian process.
        trainable_noise_variance : bool, optional
            Whether to treat the noise variance as a trainable parameter,
            by default False.
        jitter : float, optional
            The jitter term to ensure numerical stability of the covariance
            matrix, by default 1e-6.
        """
        self.kernel = kernel
        noise_variance = jnp.asarray(noise_variance)
        if trainable_noise_variance:
            self.noise_variance = NoiseParameter(noise_variance)  # type: ignore
        else:
            self.noise_variance = nnx.Variable(noise_variance)  # type: ignore

        self.jitter = jitter

    def __str__(self) -> str:
        """
        Print a string representation of the particle.

        Returns
        -------
        str
            A string representation of the particle.
        """
        variance = f"Noise Variance : {self.noise_variance.value}"
        kernel = self.kernel.__str__()

        return f"{variance}\n{kernel}"

    def display(self) -> None:
        """
        Display the particle.
        """
        print(self.__str__())

    def predictive_distribution(
        self,
        xpredict: Float[Array, " D"],
        data: Dataset,
        latent: bool = False,
    ) -> Distribution:
        """
        Calculate the predictive distribution of the Gaussian process
        for the particle.

        Inputs will be the points to predict and the training data that
        the GP gets conditioned on.

        The distribution returned will be a MultivariateNormalFullCovariance
        distribution from `tensorflow_probability.substrates.jax.distributions`, see
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance

        If `latent` is True, the predictive distribution of the latent
        function is returned, i.e. the distribution of the function
        values without the observational noise. If False, the predictive
        distribution of the full data-generating model is returned, which
        includes the observational noise


        Parameters
        ----------
        xpredict : Float[Array, " D"]
            The points to predict, as a 1D array.
        data : Dataset
            The training data that the GP is conditioned on,
            must be a `Dataset` instance from `gallifrey.data`.
        latent : bool, optional
            Whether to return the predictive distribution of the latent
            function (without observational noise), by default False.

        Returns
        -------
        Distribution
            A tensorflow probability distribution object representing
            the predictive distribution of the Gaussian process. (Specifically,
            a MultivariateNormalFullCovariance distribution from
            `tensorflow_probability.substrates.jax.distributions`).
        """

        # unpack conditioning data
        xtrain, ytrain = data.x, data.y

        # the points to be predicted, as correct type
        t = jnp.asarray(xpredict)

        # calculate covariance matrix (Σ = Kxx + Io²) for conditioning data
        Kxx = self.kernel.gram(xtrain)
        Sigma = calculate_covariance_matrix(
            Kxx,
            self.noise_variance.value,
            self.jitter,
        )

        # calculate gram and cross_covariance for prediction points
        Ktt = self.kernel.gram(t)
        Kxt = self.kernel.cross_covariance(xtrain, t)

        # solve for Σ⁻¹Kxt | [len(xtrain),len(t)] using cholesky decomposition
        cho_fac = linalg.cho_factor(Sigma)
        Sigma_inv_Kxt = linalg.cho_solve(cho_fac, Kxt)

        # calculate predictive mean  Ktx (Kxx + Io²)⁻¹y (assumes zero mean function)
        predictive_mean = jnp.matmul(Sigma_inv_Kxt.T, ytrain)

        # calculate latent covariance function  Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt
        latent_covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        latent_covariance += jnp.eye(latent_covariance.shape[0]) * self.jitter

        # The covariance matrix that we've calculated so far, is the covariance
        # of the latent distribution as estimated by the Gaussian process. To
        # get the full covariance matrix of the model (for the data generating
        # process), we need to add the noise observational noise variance.
        if latent:
            predictive_covariance = latent_covariance
        else:
            # add noise variance to the diagonal of the covariance matrix
            predictive_covariance = (
                latent_covariance
                + jnp.eye(latent_covariance.shape[0]) * self.noise_variance
            )

        return MultivariateNormalFullCovariance(
            predictive_mean,
            predictive_covariance,
        )


def transform_particle_parameters(
    particle_state: nnx.State,
    kernel_prior: KernelPrior,
    inverse: ScalarBool = False,
) -> nnx.State:
    """
    Transform parameter of a particle state (kernel parameters
    and noise variance) based on the support mapping and bijectors.

    This function is primarily used to transform the parameters
    between a constrained and unconstrained space.

    Parameters
    ----------
    particle_state : nnx.State
        The original particle state.
    kernel_prior : KernelPrior
        The kernel prior that contains the support mapping and bijectors.
    inverse : ScalarBool, optional
        If True, the inverse transformation is applied, by default False.

    Returns
    -------
    nnx.State
        The particle state with transformed parameters.
    """
    # get parameters and state
    num_parameter_array = kernel_prior.parameter_prior.num_parameter_array
    max_leaves = kernel_prior.parameter_prior.max_leaves
    max_atom_parameters = kernel_prior.parameter_prior.max_atom_parameters
    support_mapping_array = kernel_prior.parameter_prior.support_mapping_array
    support_bijectors = kernel_prior.support_bijectors

    kernel_state = particle_state.kernel
    noise_variance = jnp.array(particle_state.noise_variance.value)  # type: ignore

    # transform the noise standard deviation (using softplus bijector)
    transformed_noise_variance = lax.cond(
        inverse,
        SOFTPLUS_BIJECTOR.inverse,
        SOFTPLUS_BIJECTOR.forward,
        noise_variance,
    )

    # transform the kernel parameters
    transformed_kernel_state = transform_kernel_parameters(
        kernel_state,
        num_parameter_array,
        max_leaves,
        max_atom_parameters,
        support_mapping_array,
        support_bijectors,
        inverse,
    )

    # create a new state with the transformed parameters
    new_state = deepcopy(particle_state)
    new_state.kernel = transformed_kernel_state  # type: ignore
    new_state.noise_variance = (  # type: ignore
        particle_state.noise_variance.replace(
            transformed_noise_variance,
        )  # type: ignore
    )

    return new_state
