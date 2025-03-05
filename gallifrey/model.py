from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import PosixPath

import beartype.typing as tp
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax import jit, pmap
from jax import tree_util as jtu
from jax.scipy.special import logsumexp
from jaxtyping import Float, PRNGKeyArray, PyTree
from tensorflow_probability.substrates.jax.distributions import (
    Categorical,
    Distribution,
    InverseGamma,
    MixtureSameFamily,
)

from gallifrey.gpconfig import GPConfig
from gallifrey.data import Dataset
from gallifrey.inference.hmc import create_hmc_sampler_factory
from gallifrey.inference.rejuvenation import rejuvenate_particle
from gallifrey.inference.smc import smc_loop
from gallifrey.inference.state import GPState
from gallifrey.inference.transforms import LinearTransform, Transform
from gallifrey.kernels.library import KernelLibrary
from gallifrey.kernels.prior import KernelPrior
from gallifrey.particles import Particle
from gallifrey.utils.typing import Array, ScalarFloat, ScalarInt


def sample_noise_variance(
    key: PRNGKeyArray,
    data: Dataset,
    n: ScalarInt,
) -> Float[jnp.ndarray, " D"]:
    """
    Sample the noise variance. These are the noise variances
    assigned to the particles when creating the model without
    set noise_variance. The sample function is an
    InverseGamma distribution, with concentration = n/2 and
    scale = 1.0.

    NOTE: This is specifically not the same as the noise prior
    distribution, which is an InverseGamma(1, 1) distribution.
    The distribution here is chosen to increase the initial
    acceptance rate of the MCMC sampler.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    data : Dataset
        A dataset object containing the input x and output y.
    n : ScalarInt
        Number of samples to draw.
    Returns
    -------
     Float[jnp.ndarray, " D"]
        The sampled noise variance.
    """

    inv_gamma = InverseGamma(concentration=jnp.array(data.n / 2), scale=jnp.array(1.0))

    return jnp.array(inv_gamma.sample(seed=key, sample_shape=(n,)))


def initialize_particle_state(
    kernel_state: nnx.State,
    kernel_prior: KernelPrior,
    noise_variance: ScalarFloat | Float[jnp.ndarray, " D"],
    fix_noise: bool,
) -> tuple[nnx.GraphDef, nnx.State]:
    """
    Initialize an instance of the particle class as
    a particle with a Gaussian likelihood and the provided input
    parameters, and returns the graphdef and state of the particle.

    Parameters
    ----------
    kernel_state : nnx.State
        The kernel state.
    kernel_prior : KernelPrior
        The kernel prior object, containing various
        useful methods and attributes for the kernel
        sampling. Attributes set through config. (See
        gallifrey.kernels.prior.KernelPrior for details.)
    noise_variance : ScalarFloat | Float[jnp.ndarray, " D"]
        The noise variance.
    fix_noise : bool
        If True, the noise variance is fixed and not trainable.

    Returns
    -------
    Particle
        The initialized particle instance.
    """
    kernel = nnx.merge(kernel_prior.graphdef, kernel_state)

    particle = Particle(
        kernel=kernel,
        noise_variance=noise_variance,
        trainable_noise_variance=not fix_noise,
    )

    graphdef, state = nnx.split(particle)
    return graphdef, state


def batch_states(states: list[PyTree]) -> PyTree:
    """
    Batch a list of states (or general PyTrees) into a single
    state with batched parameters.

    This function takes a list of individual particle states
    (`states`) and transforms it into a single `nnx.State` object
    where each parameter is batched across all particles. This is useful for
    parallelized computations over particles.

    Parameters
    ----------
    states : list[nnx.State]
        List of individual particle states.

    Returns
    -------
    nnx.State
        A single state object with parameters batched across particles.
    """
    return jtu.tree_map(lambda *xs: jnp.array(xs), *states)


def unbatch_states(state: PyTree) -> list[PyTree]:
    """
    Unbatch a single state (or general PyTree) with batched parameters
    into a list of individual states.

    This function takes a single `nnx.State` object where each parameter is
    batched across all particles and transforms it into a list of individual particle
    states.

    Parameters
    ----------
    state : nnx.State
        A single state object with parameters batched across particles.

    Returns
    -------
    list[nnx.State]
        List of individual particle states.
    """
    number_of_states = len(jtu.tree_leaves(state)[0])
    return [jtu.tree_map(lambda x: x[i], state) for i in range(number_of_states)]


class GPModel:
    """
    The GP model class.

    Attributes
    ----------

    num_particles : int
        Number of particles in the model.
    config : GPConfig
        The config instance for the GP model (see
        gallifrey.config.GPConfig for details).
    kernel_prior : KernelPrior
        The kernel prior object, containing various
        useful methods and attributes for the kernel
        sampling. Attributes set through config. (See
        gallifrey.kernels.prior.KernelPrior for details.)
    noise_prior : Distribution
        The prior distribution for the noise variance, inherited
        from the config. (See gallifrey.config.GPConfig for details.)
    kernel_library : KernelLibrary
        The kernel library, containing the atomic kernels, operators,
        and prior transforms. (See gallifrey.kernels.library.KernelLibrary
        for details.)
    x :  Float[jnp.ndarray, " D"]
        The input x data.
    y :  Float[jnp.ndarray, " D"]
        The input y data.
    noise_variance:
        The initial noise variance if fixed, otherwise None.
    fix_noise : bool
        Flag indicating whether the noise variance is fixed or learned.
        This is set to True if the noise variance is provided
        as an input parameter.
    x_transform : Optional[Callable]
        A (optional) transformation applied to the input x data.
    y_transform : Optional[Callable]
        A (optional) transformation applied to the input y data.
    x_transformed :  Float[jnp.ndarray, " D"]
        The transformed input x data.
    y_transformed :  Float[jnp.ndarray, " D"]
        The transformed noise variance if fixed, otherwise None.
    noise_variance_transformed :  Float[jnp.ndarray, " D"]
        The transformed noise variance, if noise variance is provided.
    dataset : Dataset
        A dataset instance containing the transformed x and y data, processed
        for the sampler.
    particle_graphdef : nnx.GraphDef
        Graph definition for the Particle object, shared across all particles.
        This can be used to reconstruct the Particle object from a particle state.
    state : GPState
        The state of the GP model, containing the particle states and
        other relevant information. (See gallifrey.inference.state.GPState
        for details.)
    hmc_sampler_factory : Callable
        Factory function to create HMC samplers (using blackjax), configured based on
        `config.hmc_config`

    Methods
    -------
    update_state
        Update the internal gpstate of the model using a new GPState object.
    save_state
        Save a GP state to file. Note that currently only the state is saved
        and not the model itself. That means if the model is loaded with a different
        configuration (e.g. different kernel library), the state might not be
        consistent with the model.
    load_state
        Load a GP state from file. Note that currently only the state is loaded
        and not the model itself. That means if the model is loaded with a different
        configuration (e.g. different kernel library), the state might not be
        consistent with the model.
    fit_mcmc
        Fit the GP model using MCMC.
    fit_smc
        Fit the GP model using SMC.
    get_particles
        Get a list of Particle instances from a GP state. If no GP state
        is provided, the current GP state of the model is used.
    get_predictive_distributions
        Calculate the predictive distributions for the individual particles in the
        GP state. The distributions are calculated at the points `x_predict` and
        conditioned on the training data (which was supplied to construct the model
        instance).
    get_mixture_distribution
        Get the mixture distribution of an SMC state (A weighted sum of the predictive
        distributions of the individual particles). The model should be created using
        the `fit_smc` method.
    display
        Print a summary of the GP model, including particle kernels and noise variances.

    """

    def __init__(
        self,
        key: PRNGKeyArray,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
        num_particles: ScalarInt,
        noise_variance: tp.Optional[ScalarFloat] = None,
        x_transform: tp.Type[Transform] = LinearTransform,
        y_transform: tp.Type[Transform] = LinearTransform,
        config: tp.Optional[GPConfig] = None,
    ) -> None:
        """
        Initialize the GP model.

        Parameters
        ----------
        key : PRNGKeyArray
            The random key for the initial sampling.
        x : Float[Array, " D"]
            Input data, array of shape (D,).
        y : Float[Array, " D"]
            Target data, array of shape (D,).
        num_particles : ScalarInt
            The number of particles in the model.
        noise_variance : tp.Optional[ScalarFloat], optional
            The variance of the observation noise. If
            None, the noise variance is sampled and treated
            as a trainable parameter. By default None.
            NOTE: Currently heteroscadastic noise is not
            supported.
        x_transform : Transform, optional
            A transformation applied to the input x data, must
            be an instance of gallifrey.inference.transforms.Transform
            class. By default LinearTransform. (Used to normalise
            data for easier training.)
        y_transform : Optional[Callable], optional
            A transformation applied to the input y data, must
            be an instance of gallifrey.inference.transforms.Transform
            class. By default LinearTransform. (Used to normalise
            data for easier training.)
        config : tp.Optional[GPConfig], optional
            The configuration object for the GP model. Contains
            information of the kernel and parameter priors, the
            mean function, the max depth of the tree kernel, etc.
            By default None, in which case the default configuration
            is used (see gallifrey.config.GPConfig for details).
        """
        if len(x) != len(y):
            raise ValueError(
                f"Input data x and y must have the same length, "
                "but got len(x)={len(x)} and len(y)={len(y)}."
            )

        # set basic attributes
        self.config = config if config is not None else GPConfig()
        self.fix_noise = True if noise_variance is not None else False

        # create kernel prior
        kernel_library = KernelLibrary(
            atoms=deepcopy(self.config.atoms),
            operators=deepcopy(self.config.operators),
            prior_transforms=deepcopy(self.config.prior_transforms),
        )

        self.kernel_prior = KernelPrior(
            kernel_library,
            max_depth=deepcopy(self.config.max_depth),
            num_datapoints=len(x),
            probs=deepcopy(self.config.node_probabilities),
        )

        # preprocess data (apply transformations, and set attributes)
        self._preprocess_data(
            x,
            y,
            x_transform,
            y_transform,
            noise_variance,
        )

        # create a particle_graphdef attributes (using a randomly
        # initilized particle state). The graphdef attribute,
        # is used to create particle instances from the particle states
        self.particle_graphdef, particle_state = initialize_particle_state(
            self.kernel_prior.sample(jr.PRNGKey(0)),
            self.kernel_prior,
            self._sample_noise_variance(jr.PRNGKey(0)),
            self.fix_noise,
        )

        # sample initial particles and create initial GP state
        key, particle_key = jr.split(key)
        self._initialize_state(particle_key, num_particles)

        # create HMC sampler factory for rejuvenation
        self.hmc_sampler_factory = create_hmc_sampler_factory(
            self.config.hmc_config,
            int(self.kernel_prior.max_kernel_parameter + (not self.fix_noise)),
        )

    def _preprocess_data(
        self,
        x: Float[Array, " D"],
        y: Float[Array, " D"],
        x_transform: tp.Type[Transform],
        y_transform: tp.Type[Transform],
        noise_variance: tp.Optional[ScalarFloat],
    ) -> None:
        """
        Preprocesses the input data by applying transformations
        and creating a Dataset object.

        Applies the specified transformations (`x_transform` and `y_transform`)
        to the input data `x` and `y`. If a fixed `noise_variance` is provided,
        it is also transformed using `y_transform`.
        Finally, it creates a `Dataset` object with the transformed
        data, which is used for GPJax computations.


        Parameters
        ----------
        x :  Float[jnp.ndarray, " D"]
            The input x data.
        y :  Float[jnp.ndarray, " D"]
            The input y data.
        x_transform : tp.Type[Transform]
            A transformation applied to the input x data.
        y_transform : tp.Type[Transform]
            A transformation applied to the input y data.
        noise_variance : tp.Optional[ScalarFloat]
            The variance of the observation noise.
        """
        # set attributes
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)

        # create transformation from data and apply to x and y
        self.x_transform = x_transform.from_data_range(self.x, 0, 1)
        self.y_transform = y_transform.from_data_width(self.y, 1)

        self.x_transformed = self.x_transform.apply(self.x)
        self.y_transformed = self.y_transform.apply(self.y)

        # transform noise variance if provided
        if noise_variance is not None:
            self.noise_variance: ScalarFloat | None = jnp.asarray(noise_variance)
            self.noise_variance_transformed: ScalarFloat | None = jnp.asarray(
                self.y_transform.apply_var(noise_variance)
            )
        else:
            self.noise_variance = None
            self.noise_variance_transformed = None

        # ensure that x_transformed and y_transformed are still 1D arrays and of
        # the same length
        if (not self.x_transformed.ndim == 1) or (not self.y_transformed.ndim == 1):
            raise ValueError(
                "x_transformed and y_transformed must be 1D arrays. "
                "Please check the transformation functions."
            )
        if len(self.x_transformed) != len(self.y_transformed):
            raise ValueError(
                "x_transformed and y_transformed must have the same length."
            )

        self.data = Dataset(
            self.x_transformed,
            self.y_transformed,
        )

    def _sample_noise_variance(
        self,
        key: PRNGKeyArray,
    ) -> ScalarFloat:
        """
        Sample the noise variance from the prior distribution, if no
        noise variance is provided. If a noise variance is provided, it
        is returned as is.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.

        Returns
        -------
         Float[jnp.ndarray, " D"]
            The sampled noise variance, one for each particle. (If noise_variance
            is provided, the same value is returned for all particles.)
        """

        if self.noise_variance is None:
            key, noise_key = jr.split(key)
            variance: ScalarFloat = jnp.array(
                self.config.noise_prior.sample(
                    seed=noise_key,
                ),
                dtype=self.x.dtype,
            )
        else:
            assert self.noise_variance_transformed is not None
            variance = self.noise_variance_transformed
        return variance

    def _initialize_state(
        self,
        key: PRNGKeyArray,
        num_particles: ScalarInt,
    ) -> None:
        """
        Initialize the state of the GP model. This primarily
        entails creating the initial particle states by sampling
        the kernel and noise variance (if not fixed).

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.
        num_particles : ScalarInt
            Number of particles in the model.

        Returns
        -------
        GPState
            A GPState object representing the state of the GP model (see
            gallifrey.inference.state.GPState for details).
        """

        def make_particle_state(key: PRNGKeyArray) -> nnx.State:
            """
            Create partice by sampling the kernel and noise variance
            (if not fixed), and initializing the particle state.
            """
            key, kernel_key, variance_key = jr.split(key, 3)
            kernel_state = self.kernel_prior.sample(kernel_key)
            noise_variance = self._sample_noise_variance(variance_key)

            _, particle_state = initialize_particle_state(
                kernel_state,
                self.kernel_prior,
                noise_variance,
                self.fix_noise,
            )

            return particle_state

        # create particle states by looping over the number of particles,
        # and batch into a single state object
        num_particles = int(num_particles)
        particle_states = batch_states(
            [make_particle_state(key) for key in jr.split(key, num_particles)]
        )

        self.state = GPState(
            particle_states=particle_states,
            num_particles=num_particles,
            num_data_points=self.data.n,
            mcmc_accepted=jnp.zeros(num_particles),
            hmc_accepted=jnp.zeros(num_particles),
        )

    def __str__(self) -> str:
        """
        Return a string representation of the GPModel.

        Returns
        -------
        str
            A string representation of the GPModel.
        """
        return f"GPModel(num_particles={self.num_particles})"

    @property
    def kernel_library(self) -> KernelLibrary:
        """
        Get the kernel library.

        Returns
        -------
        KernelLibrary
            The kernel library.
        """
        return self.kernel_prior.kernel_library

    @property
    def noise_prior(self) -> Distribution:
        """
        Get the noise prior distribution.

        Returns
        -------
        InverseGamma
            The noise prior distribution.
        """
        return self.config.noise_prior

    @property
    def num_particles(self) -> int:
        """
        Get the number of particles in the model.

        Returns
        -------
        int
            The number of particles.
        """
        return self.state.num_particles

    def update_state(self, gpstate: GPState) -> GPModel:
        """
        Update the GP state. Returns a new GPModel instance (no in-place
        update). If the number of particles in the new state is different
        from the current state, the num_particles attribute is updated.

        Note that no other attributes are updated. If the particle states
        were created using a different configuration, the model will not be
        consistent.

        Parameters
        ----------
        gpstate : GPState
            The new GP state to update the model with.

        Returns
        -------
        GPModel
            A new GPModel instance with updated state.

        """

        new_gpmodel = deepcopy(self)
        new_gpmodel.state = gpstate

        return new_gpmodel

    def save_state(
        self,
        path: str | PosixPath,
        gpstate: tp.Optional[GPState] = None,
    ) -> None:
        """
        Save a GP state to file.

        Note that only the state is saved and not the model itself.
        That means if the model is loaded with a different configuration
        (e.g. different kernel library), the state might not be consistent
        with the model.

        TODO: Implement saving the model configuration as well.

        Parameters
        ----------
        path : str | PosixPath
            The path where to save the GP state, must
            be an absolute path.
        gpstate : tp.Optional[GPState], optional
            The GP state to save, If None, the current state
            of the model is used. By default None.

        """
        if gpstate is None:
            gpstate = self.state

        with open(path, "wb") as file:
            pickle.dump(gpstate, file)

        return None

    @classmethod
    def load_state(
        cls,
        path: str | PosixPath,
    ) -> GPState:
        """
        Load a GP state from file.

        Note that only the state is loaded. It is assumed that the model
        configuration is consistent with how the state was saved. If the
        model configuration is different, the loaded state might not be
        consistent with the model.

        The model does not get update with the loaded state. To update the
        model, use the `update_gpstate` method with the loaded state.

        TODO: Implement loading the model configuration as well.

        Parameters
        ----------
        path : str | PosixPath
            The path where to load the GP state from.

        Returns
        -------
        GPState
            An instance of the GPState object, containing the loaded state.

        """
        with open(path, "rb") as file:
            gpstate = pickle.load(file)
        return gpstate

    def fit_mcmc(
        self,
        key: PRNGKeyArray,
        n_mcmc: ScalarInt,
        n_hmc: ScalarInt,
        verbosity: ScalarInt = 0,
    ) -> tuple[GPState, GPState]:
        """
        Fits the GP model using MCMC.
        It perfoms n_mcmc iterations of the structure, and for each
        accepted structure move performs n_hmc iterations of the HMC
        sampler over the parameters.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for MCMC sampling.
        n_mcmc : ScalarInt
            Number of MCMC iterations over kernel structure.
        n_hmc : ScalarInt
            Number of HMC steps for continuous parameters. Only used if
            the structure move is accepted.
        verbosity : ScalarInt, optional
            The verbosity level, by default 0. Debugging information
            is printed if `verbosity > 1`.

        Returns
        -------
        GPState
            The final state of the model, wrapped in an GPState object (see
            gallifrey.inference.state.GPState for details).
        GPState
            The history over all MCMC iterations, wrapped in an GPState object.

        """
        if not isinstance(n_mcmc, int):
            raise TypeError(
                f"Expected `n_mcmc` to be an integer, but got {type(n_mcmc)}."
            )
        if not isinstance(n_hmc, int):
            raise TypeError(
                f"Expected `n_hmc` to be an integer, but got {type(n_hmc)}."
            )
        if n_mcmc <= 0:
            raise ValueError(
                f"Expected `n_mcmc` to be a positive integer, but got {n_mcmc}."
            )
        if n_hmc <= 0:
            raise ValueError(
                f"Expected `n_hmc` to be a positive integer, but got {n_hmc}."
            )

        def wrapper(
            key: PRNGKeyArray,
            state: nnx.State,
        ) -> tuple[nnx.State, nnx.State, ScalarInt, ScalarInt]:
            """Wrapper around the rejuvenate_particle function using
            GPmodel attributes."""
            return rejuvenate_particle(
                key,
                state,
                self.data,
                self.kernel_prior,
                self.noise_prior,
                n_mcmc=n_mcmc,
                n_hmc=n_hmc,
                fix_noise=self.fix_noise,
                hmc_sampler_factory=self.hmc_sampler_factory,
                verbosity=verbosity,
            )

        final_state, history, accepted_mcmc, accepted_hmc = pmap(
            jit(wrapper), in_axes=0
        )(
            jr.split(key, self.num_particles),
            self.state.particle_states,  # use states batched over 0th axis
        )

        # print information
        if verbosity > 0:
            for i, acc_mcmc, acc_hmc in zip(
                range(self.num_particles), accepted_mcmc, accepted_hmc
            ):
                print(
                    f"Particle {i+1} | Accepted: MCMC[{acc_mcmc}/{n_mcmc}] "
                    f" HMC[{acc_hmc}/{acc_mcmc*n_hmc}]"
                )

        # wrap final state and history in GPState objects, for consistency
        # with the SMC algorithm
        final_state_wrapped = GPState(
            particle_states=final_state,
            num_particles=self.num_particles,
            num_data_points=self.data.n,
            mcmc_accepted=accepted_mcmc,
            hmc_accepted=accepted_hmc,
        )

        history_wrapped = GPState(
            particle_states=history,
            num_particles=self.num_particles,
            num_data_points=self.data.n,
            mcmc_accepted=accepted_mcmc,
            hmc_accepted=accepted_hmc,
        )

        return final_state_wrapped, history_wrapped

    def fit_smc(
        self,
        key: PRNGKeyArray,
        annealing_schedule: tuple[int, ...],
        n_mcmc: ScalarInt,
        n_hmc: ScalarInt,
        verbosity: int = 0,
    ) -> tuple[GPState, GPState]:
        """
        Fits the GP model using SMC.

        For a detailed description of the SMC algorithm, see the
        'gallofrey.inference.smc.smc_loop' function.

        Parameters
        ----------
        key : PRNGKeyArray
            The random key for the SMC sampling.
        annealing_schedule : tuple[int, ...]
            The data annealing schedule for the SMC algorithm,
            number of data points to consider at each step.
            NOTE: Must be given in form of a tuple of integers, for
            jax compatibility. Easily generated using the `generate`
            method of the `Schedule` in `gallifrey.schedule`.
        n_mcmc : ScalarInt
            Number of MCMC iterations over kernel structure per
            SMC step.
        n_hmc : ScalarInt
            Number of HMC steps for continuous parameters per
            SMC step. Only used if the structure move is accepted.
        verbosity : int, optional
            The verbosity level, by default 0. Debugging information
            is printed if `verbosity > 1`.

        Returns
        -------
        GPState
            The final SMC state. Contains the final particle states
            and the final weights (among other things, see
            'gallifrey.inference.state.GPState' for details).
        GPState
            The history of the SMC algorithm. Contains the particle
            states and weights at each step of the algorithm.

        """

        final_smc_state, history = smc_loop(
            key,
            self.state.particle_states,
            annealing_schedule,
            int(self.num_particles),
            self.data,
            self.kernel_prior,
            self.noise_prior,
            self.fix_noise,
            self.hmc_sampler_factory,
            n_mcmc,
            n_hmc,
            verbosity=verbosity,
        )

        batched_history: GPState = batch_states(history)

        return final_smc_state, batched_history

    def get_particles(
        self,
        gpstate: tp.Optional[GPState] = None,
    ) -> list[Particle]:
        """
        Get a list of Particle instances from a GP state.

        If no state is provided, the current state of the model
        is used.

        Parameters
        ----------
        gpstate : tp.Optional[GPState], optional
            The GP state to extract the particles from. If None,
            the current state of the model is used. By default None.

        Returns
        -------
        list[Particle]
            A list of Particle instances, corresponding to the
            individual states.
        """
        if gpstate is None:
            gpstate = self.state

        unbatched_particle_states = unbatch_states(gpstate.particle_states)
        return [
            nnx.merge(self.particle_graphdef, state)
            for state in unbatched_particle_states
        ]

    def get_predictive_distributions(
        self,
        xpredict: Float[Array, " D"],
        data: tp.Optional[Dataset] = None,
        gpstate: tp.Optional[GPState] = None,
        latent: bool = False,
    ) -> list[Distribution]:
        """
        Calculate the predictive distributions for the individual particles
        in the GP state. The distributions are calculated at the points
        `x_predict` and conditioned on the training data (which was supplied
        to construct the model instance).

        The distributions are returned as a list of tensorflow probability
        distribution objects, see
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance

        If no state is provided, the current state of the model
        is used.

        If `latent` is True, the predictive distribution is that of the latent
        function, i.e. the distribution of the function values without the
        observational noise. If False, the predictive distribution of the full
        data-generating model is returned, which includes the observational noise.

        Parameters
        ----------
        xpredict : Float[Array, " D"]
            The points to predict, as a 1D array.
        data : tp.Optional[Dataset], optional
            The data to condition the predictive distribution on. If None,
            the training data of the model is used. By default None.
        gpstate : tp.Optional[GPState], optional
            The GP state object, containing the particle states. If None,
            the current state of the model is used. By default None.
        latent : bool, optional
            Whether to return the predictive distribution of the latent
            functions only (without observational noise), by default
            False.

        Returns
        -------
        list[Distribution]
            A list of tensorflow probability distribution objects
            representing the predictive distributions of the Gaussian
            processes. (Specifically, a MultivariateNormalFullCovariance
            distribution from `tensorflow_probability.substrates.jax.distributions`).

        """
        gpstate = self.state if gpstate is None else gpstate
        data = self.data if data is None else data

        particles = self.get_particles(gpstate)
        distributions = [
            particle.predictive_distribution(
                jnp.atleast_1d(xpredict).squeeze(),
                data,
                latent,
            )
            for particle in particles
        ]

        return distributions

    def get_mixture_distribution(
        self,
        xpredict: Float[Array, " D"],
        gpstate: tp.Optional[GPState] = None,
        data: tp.Optional[Dataset] = None,
        log_weights: tp.Optional[Float[Array, " N"]] = None,
        num_particles: tp.Optional[int] = None,
        key: tp.Optional[PRNGKeyArray] = None,
        latent: bool = False,
    ) -> Distribution:
        """
        Get the mixture distribution of an SMC state.

        The predictive distributions for an SMC ensemble are
        the individual predictive (Gaussion) distributions of the
        particles, weighted by the particle weights. The resulting
        distribution is a Gaussian mixture model, implemented as
        a `MixtureSameFamily` distribution from `tensorflow_probability`, see
        https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily

        The mixture distribution is calculated at the points `xpredict`
        and conditioned on the training data (which was supplied to construct
        the model instance).

        The input should be an GPState object as returned by the `fit_smc` method.
        Alternatively, the log weights can be provided explicitly using the
        `log_weights` argument.


        Parameters
        ----------
        xpredict : Float[Array, " D"]
            The points to predict, as a 1D array.
        gpstate : tp.Optional[GPState], optional
            The GP state object, containing the particle states and log weights for
            the mixture distribution. If None, the current state of the model is used.
            By default None.
        data : tp.Optional[Dataset], optional
            The data to condition the predictive distribution on. If None,
            the training data of the model is used. By default None.
        log_weights : Float[Array, " N"], optional
            The log weights of the particles. If None, the log weights from the
            GP state are used. An error is raised if the log weights are not provided
            and the GP state does not contain log weights. By default None.
        num_particles : tp.Optional[int], optional
            Number of particles to include in the mixture distribution. If None,
            all particles in the state are included. If provided, a random sample
            of particles is chosen based on the weights. By default None.
        key : tp.Optional[PRNGKeyArray], optional
            Random key for sampling the particles. Required if `num_particles` is
            provided. By default None.
        latent : bool, optional
            Whether to return the predictive distribution of the latent
            functions only (without observational noise), by default
            False.

        Returns
        -------
        Distribution
            A tensorflow probability distribution object representing
            the Gaussian mixture distribution of the Gaussian processes
            (a MixtureSameFamily distribution).

        Raises
        ------
        ValueError
            If the GPState object contains no log weights and `log_weights` is None.
        ValueError
            If the number of particles and log weights are inconsistent.

        """
        gpstate = self.state if gpstate is None else gpstate
        data = self.data if data is None else data

        if (gpstate.log_weights is None) and (log_weights is None):
            raise ValueError(
                "The GPState object contains no log weights. This might be "
                "because the state was produced with the MCMC sampler or the "
                "initial state of the GPModel was used. Please either run "
                "`fit_smc` or provide the log weights explicitly. (Note: If "
                "you already ran `fit_smc` and this error occurs, make sure to "
                "passed the the output of `fit_smc` to this method, or run the "
                "`update_state` method with the output of `fit_smc` as input.)"
            )

        log_weights = log_weights if log_weights is not None else gpstate.log_weights

        assert log_weights is not None
        if len(log_weights) != self.num_particles:
            raise ValueError(
                f"Inconsistent number of particles and log weights, "
                f"expected {self.num_particles} but got {len(log_weights)}."
            )

        individual_distributions = self.get_predictive_distributions(
            xpredict,
            data,
            gpstate,
            latent,
        )

        if num_particles is not None:
            if key is None:
                raise ValueError(
                    "If `num_particles` is provided, `key` must also be provided. "
                    "This is the random key for sampling from the particles."
                )
            if num_particles > self.num_particles:
                raise ValueError(
                    f"Number of particles to sample ({num_particles}) "
                    f"exceeds the total number of particles ({self.num_particles})."
                )
            # choose random sample of particles based on weights
            weights = jnp.exp(log_weights)
            particle_indices = jr.choice(
                key,
                jnp.arange(len(weights)),
                p=weights,
                shape=(num_particles,),
                replace=False,
            )
            # normalize weights
            log_weights = log_weights[particle_indices] - logsumexp(
                log_weights[particle_indices]
            )
            # select particles
            individual_distributions = [
                individual_distributions[idx] for idx in particle_indices
            ]

        batched_distributions = batch_states(individual_distributions)

        mixture_model = MixtureSameFamily(
            mixture_distribution=Categorical(logits=log_weights),
            components_distribution=batched_distributions,
        )

        return mixture_model

    def display(
        self,
        gpstate: tp.Optional[GPState] = None,
        num_particles: tp.Optional[int] = None,
    ) -> None:
        """
        Prints a summary of the GP model, including particle kernels
        and noise variances.

        If no particle states are provided, the current batched state of the model
        is used.

        Iterates through each particle, merges its graph definition and
        state, and prints the particle index, noise variance, and the kernel structure.
        Useful for inspecting the current state of the particle ensemble.

        Parameters
        ----------
        gpstate : tp.Optional[GPState], optional
            The GP state object, containing the particle states. If None, the current
            state of the model is used. By default None.
        num_particles : tp.Optional[int], optional
            Number of particles to display. If None, all particles are displayed.
            By default None.

        """
        gpstate = gpstate if gpstate is not None else self.state
        num_particles = self.num_particles if num_particles is None else num_particles

        particles = self.get_particles(gpstate)
        for i in range(num_particles):
            print("=" * 50)
            if gpstate.log_weights is not None:
                print(
                    f"Particle {i+1} "
                    f"| Weight: {jnp.exp(gpstate.log_weights[i]):.2f} "
                    f"| Variance: {particles[i].noise_variance.value} "
                )
            else:
                print(
                    f"Particle {i+1} "
                    f"| Variance: {particles[i].noise_variance.value} "
                )
            print(f"{particles[i].kernel}")
