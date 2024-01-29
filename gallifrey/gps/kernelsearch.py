import warnings
from copy import deepcopy
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from beartype.typing import Callable
from gpjax.fit import FailedScipyFitError
from jax import jit, tree_map
from jax.flatten_util import ravel_pytree
from jax.stages import Wrapped
from jaxtyping import Array, install_import_hook
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm import tqdm

from gallifrey.gps.description import _describe_kernel
from gallifrey.gps.kernels import (
    CombinationKernel,
    ProductKernel,
    SumKernel,
    flatten_kernels,
)
from gallifrey.gps.trainables import (
    get_trainables,
    set_trainables,
    set_obs_stddev,
)
from gallifrey.util import tqdm_joblib

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# random seed
key = jr.PRNGKey(42)


class Node:
    def __init__(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
        parent: Optional["Node"] = None,
    ) -> None:
        """Node of the search tree, containing the posterior
        model.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
        max_log_likelihood : Optional[float], optional
             The log likelihood found after fittng, by default None
        parent : Optional[Node], optional
            Parent node in the search tree, by default None
        """
        self.children: list["Node"] = []
        self.parent = parent

        self.update(
            posterior,
            max_log_likelihood,
        )

    def __repr__(self) -> str:
        return f"Model with kernel: {_describe_kernel(self.posterior)}"

    def _describe_kernel(self) -> str:
        """
        Generate string description of current kernel. Works with nested
        CombinationKernels in the kernel tree, but only those created
        explicity be the kernel search and its particular structure.

        Parameters
        ----------
        kernel : AbstractKernel
            The kernel.

        Returns
        -------
        str
            String description of kernel.
        """
        return _describe_kernel(self.posterior)

    def get_trainables(self, unconstrain: bool = False) -> Array:
        """Print values of trainable parameter
        of model.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
                The gpjax posterior model.
        unconstrain : bool, optional
            If True, model parameter are pushed trough bijector first
            making their support encompass the entire real line, by
            default False

        Returns
        -------
        Array
            List of trainable parameter values.
        """
        return get_trainables(self.posterior, unconstrain)

    def update_obs_stddev(self, obs_stddev: float | Array) -> None:
        """Update psoterior with new obs_stddev.

        Parameters
        ----------
        obs_stddev : float | Array
            The new obs_stddev.
        """
        self.posterior: gpx.gps.AbstractPosterior = set_obs_stddev(
            self.posterior, obs_stddev
        )

    def update_trainables(
        self,
        parameter: tuple | list | Array | NDArray,
        unconstrain: bool = False,
    ) -> None:
        """Update posterior with new trainable parameter.

        Parameters
        ----------
        parameter : tuple | list | Array | NDArray
            The list of new parameter. Must be of same length
            as the number of trainable parameter
        unconstrain : bool, optional
            If True, model parameter are pushed trough bijector first
            making their support encompass the entire real line. The given
            parameter list must then be in this unconstrained space, by
            default False

        """
        self.posterior = set_trainables(self.posterior, parameter, unconstrain)

    def update(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
    ) -> None:
        """Update the node by setting new posterior,
        max_log_likelihood and n_posterior.
        Automatically calculates n_parameter from the model,
        AIC, and BIC if max_log_likelihood is available.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
        max_log_likelihood : Optional[float], optional
             The log likelihood found after fittng, by default None
        parent : Optional[Node], optional
        """
        self.posterior = posterior
        self.n_datapoints = posterior.likelihood.num_datapoints
        self.n_parameter = sum(ravel_pytree(posterior.trainables())[0])
        if max_log_likelihood is None:
            self.max_log_likelihood = -jnp.inf
        else:
            self.max_log_likelihood = max_log_likelihood

        self.aic = float(self.n_parameter * 2 - 2 * self.max_log_likelihood)
        self.bic = float(
            self.n_parameter * jnp.log(self.n_datapoints) - 2 * self.max_log_likelihood
        )

    def _add_child(
        self,
        node: "Node",
    ) -> None:
        """Add node to list of children.

        Parameters
        ----------
        node : Node
            Node instance of child in search tree.
        """
        self.children.append(node)


class KernelSearch:
    def __init__(
        self,
        kernel_library: list[gpx.kernels.AbstractKernel],
        X: NDArray | Array,
        y: NDArray | Array,
        obs_stddev: float | Array,
        fit_obs_stddev: bool = False,
        objective: str | gpx.objectives.AbstractObjective | Wrapped = "mll",
        criterion: str = "aic",
        likelihood: Optional[gpx.likelihoods.AbstractLikelihood] = None,
        mean_function: Optional[gpx.mean_functions.AbstractMeanFunction] = None,
        root_kernel: Optional[gpx.kernels.AbstractKernel] = None,
        fitting_mode: str = "scipy",
        num_iters: int = 1000,
        num_threads: int = 1,
        verbosity: int = 1,
        clear_caches: bool = True,
    ):
        """Greedy search for optimal Gaussian process kernel structure.

        Parameters
        ----------
        kernel_library : list[gpx.kernels.AbstractKernel]
            List of (initialized) kernel instances that will be used to grow the tree.
        X : NDArray | Array
            Array of X data.
        y : NDArray | Array
            Array of y data. The array can be 2d with dimensions (num_datapoints,
            num_datasets), in which case the data along the second axis is
            assumed to be independent datasets. In this case, the best kernel
            that describes all datasets is determined.
        obs_stddev : float | Array
            The standard deviation of the y data. Is ignored if custom
            likelihood is given. Must be of same length as the number of datasets.
        fit_obs_stddev : bool, optional
            Wether to estimate the standard deviation during fitting process. Is
            ignored if custom likelihood is given, by default False
        objective : str | gpx.objectives.AbstractObjective | Wrapped, optional
            The objective function used to evalute the quality of a fit. If string,
            must be either 'mll' for the marginal log likelihood, or 'loocv' for
            leave-out-out predictive log probability. Otherwise must be a function
            takes posterior and train_data as inputs and returns a scalar probability,
            by default 'mll'.
        criterion : str, optional
            Criterion to determine quality of fit. Choose from "aic" or "bic",
            by default "aic".
        likelihood : Optional[gpx.likelihoods.AbstractLikelihood], optional
            Function that calculates the likelihood of the model. By default None,
            which defaults to the Gaussian likelihood with standard deviation
            given by obs_stddev.
        mean_function : Optional[gpx.mean_functions.AbstractMeanFunction], optional
            The mean function of the Gaussian process. By default None, which
            sets the mean to zero.
        root_kernel : gpx.kernels.AbstractKernel
            Optional kernel instance that is used as root for the search tree. By
            default None, which uses the kernel library as roots.
        fitting_mode : str, optional
            Fitting procedure. Choose between "scipy" and "adam", by
            default "scipy".
        num_iters : int, optional
            (Maximum) number of iterations for the fitting, by default 1000.
        num_threads : int, optional
            Number of parallel threads for tree expansion. Deactivate parallelisation
            using num_threads=1, by default 1.
        verbosity : int, optional
            Verbosity of the output between 0 and 2, by default 1
        clear_caches: bool, optional
            If True, clear jax jit caches when building the tree. Helps to avoid
            certain tracing errors, by default True.
        """
        if clear_caches:
            jax.clear_caches()
        self.clear_caches = clear_caches

        if isinstance(obs_stddev, float):
            obs_stddev = jnp.array(obs_stddev)
        assert isinstance(obs_stddev, Array)
        if obs_stddev.ndim == 0:
            obs_stddev = obs_stddev.reshape(-1)
        self.obs_stddev = obs_stddev

        # set defaults
        if likelihood is None:
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=len(X))
            likelihood = likelihood.replace_trainable(
                obs_stddev=fit_obs_stddev  # type: ignore
            )  # type: ignore
        if mean_function is None:
            mean_function = gpx.mean_functions.Zero()
            # if not jnp.isclose(jnp.mean(y), 0):
            #    raise ValueError(
            #        "If no mean function is given, data must be centered at 0. "
            #        "Please center data first using y - mean(y)."
            #    )
        if isinstance(objective, str):
            if objective.lower() == "mll":
                objective = jit(gpx.objectives.ConjugateMLL(negative=True))
            elif objective.lower() == "loocv":
                objective = jit(gpx.objectives.ConjugateLOOCV(negative=True))
            else:
                raise ValueError("If objective is a string, must be 'mll' or 'loocv'.")

        self.likelihood = likelihood
        self.objective = objective
        self.criterion = criterion.lower()

        # match dimensions of X and y
        self.X = jnp.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
        self.X = self.X.T if self.X.shape[1] > self.X.shape[0] else self.X
        self.y = jnp.asarray(y).reshape(-1, 1) if y.ndim == 1 else y
        self.y = (
            self.y.T
            if (self.X.shape[0] == self.y.shape[1])
            or (self.X.shape[1] == self.y.shape[0])
            else self.y
        )
        if not any([a == b for a, b in zip(self.X.shape, self.y.shape)]):
            raise ValueError("X and y must match along one axis.")

        if len(self.obs_stddev) != self.y.shape[1]:
            raise ValueError(
                "There must be as many values in obs_stddev as there are datasets in y."
            )

        self.kernel_library = kernel_library
        self.nodes: list[Node] = []

        self.fitting_mode = fitting_mode
        self.num_iters = num_iters
        self.num_threads = num_threads

        self.verbosity = verbosity

        # create root node
        self.root = [
            Node(
                likelihood
                * gpx.gps.Prior(
                    mean_function=mean_function,
                    kernel=kernel,
                )
            )
            for kernel in (kernel_library if root_kernel is None else [root_kernel])
        ]

    def get_criterion(self, node: Node) -> float:
        """Return chosen value of fitting
        criterion.

        Parameters
        ----------
        node : Node
            The node for which to return the value.

        Returns
        -------
        float
            The fitting criterion value (AIC or BIC).
        """
        if self.criterion == "aic":
            return node.aic
        elif self.criterion == "bic":
            return node.bic
        else:
            raise ValueError("criterion must be 'aic' or 'bic'.")

    def get_fit_function(
        self,
    ) -> Callable:
        """Returns callable fit function, that fits the hyperparameter
        to of a posterior model to input data x and y. The Callable
        returns the optimized posterior and the maximum log likelihood.
        Can be done using scipy's 'minimize' function using the 'adam'
        optimiser from optax.

        Returns
        -------
        Callable
            Returns the fitting function, with signature
            fit_function(posterior, X, y) ->
            optimised_posterior, max_log_likelihood.

        Raises
        ------
        ValueError
            Thrown if optimiser mode is unknown.
        """

        if self.fitting_mode == "scipy":

            def fit_function(
                posterior: gpx.gps.AbstractPosterior,
                X: NDArray | Array,
                y: NDArray | Array,
            ) -> tuple[Callable, float]:
                """Fit function using scipy optimizer."""
                data = gpx.Dataset(X=X, y=y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    try:
                        optimized_posterior, history = gpx.fit_scipy(
                            model=posterior,
                            objective=self.objective,  # type: ignore
                            train_data=data,
                            max_iters=self.num_iters,
                            verbose=self.verbosity >= 2,
                        )
                    except FailedScipyFitError:
                        optimized_posterior, history = posterior, [jnp.inf]
                return optimized_posterior, float(
                    jnp.nan_to_num(-history[-1], nan=-jnp.inf)
                )

        elif self.fitting_mode == "adam":

            def fit_function(
                posterior: gpx.gps.AbstractPosterior,
                X: NDArray | Array,
                y: NDArray | Array,
            ) -> tuple[Callable, float]:
                """Fit function using 'adam' optimizer."""
                data = gpx.Dataset(X=X, y=y)

                static_tree = tree_map(lambda x: not x, posterior.trainables)
                optim = ox.chain(
                    ox.adamw(learning_rate=3e-4),
                    ox.masked(
                        ox.set_to_zero(),
                        static_tree,
                    ),
                )
                optimized_posterior, history = gpx.fit(
                    model=posterior,
                    objective=self.objective,  # type: ignore
                    train_data=data,
                    optim=optim,
                    key=key,
                    num_iters=self.num_iters,
                    verbose=self.verbosity >= 2,
                )
                return optimized_posterior, float(
                    jnp.nan_to_num(-history[-1], nan=-jnp.inf)
                )

        else:
            raise ValueError("'fitting_mode' must be 'scipy' or 'adam'.")

        return fit_function

    def expand_node(self, node: Node) -> None:
        """Logic to expand a node in the search tree. The
        child nodes are created by adding or multiplying
        the kernels from the kernel library to the current
        kernel. New kernels are added to children of
        the Node instance.

        Parameters
        ----------
        node : Node
            Current Node instance to expand.
        """
        for kernel_operation in [ProductKernel, SumKernel]:
            for ker in self.kernel_library:
                kernel = deepcopy(node.posterior.prior.kernel)
                new_kernel = deepcopy(ker)  # type: ignore

                if kernel_operation == SumKernel:
                    # create new additive term
                    composite_kernels = [
                        kernel_operation(kernels=[kernel, new_kernel])
                    ]  # type: ignore

                elif kernel_operation == ProductKernel:
                    # further kernels have variance fixed, so that we only
                    # have one multiplicative constant per term,
                    try:
                        new_kernel = new_kernel.replace_trainable(
                            variance=False  # type: ignore
                        )
                    except ValueError:
                        pass

                    # multiply each term inidivually, if current kernel
                    # already is a sum,
                    # there is a special check to make sure there are not multiple
                    # white kernels in a term, since that doesn't change anything
                    composite_kernels = []
                    if (
                        isinstance(kernel, CombinationKernel)
                        and kernel.operator == jnp.sum
                    ):
                        terms = kernel.kernels
                        assert terms
                        for i in range(len(terms)):
                            new_terms = deepcopy(terms)

                            try:
                                kernel_names = [
                                    k.name
                                    for k in flatten_kernels(
                                        new_terms[i].kernels  # type: ignore
                                    )
                                ]
                            except AttributeError:
                                kernel_names = [new_terms[i].name]

                            if (
                                new_kernel.name != "White"
                                or "White" not in kernel_names
                            ):
                                new_terms[i] = kernel_operation(
                                    kernels=[new_terms[i], new_kernel]
                                )
                                composite_kernels.append(SumKernel(kernels=new_terms))

                    else:
                        if kernel.name != "White" or new_kernel.name != "White":
                            composite_kernels.append(
                                kernel_operation(kernels=[kernel, new_kernel])
                            )
                else:
                    raise RuntimeError

                for composite_kernel in composite_kernels:
                    new_prior = gpx.gps.Prior(
                        mean_function=node.posterior.prior.mean_function,
                        kernel=composite_kernel,
                    )

                    new_posterior = self.likelihood * new_prior
                    node._add_child(Node(new_posterior, parent=node))

    def compute_layer(
        self,
        layer: list[Node],
        current_depth: int,
    ) -> None:
        """Fit the hyperparameter of the posterior for
        all nodes in the current layer.
        If y is a 2d array of multiple datasets, the total
        likelihood (product of each likelihood) is used for
        determining the best model.

        Parameters
        ----------
        layer : list[Node]
            List of nodes in the current layer.
        current_depth : int
            Integer depth of current layer, used for
            tracking.
        """
        fit_function = self.get_fit_function()

        def inner_loop(
            posterior: gpx.gps.AbstractPosterior,
        ) -> tuple[gpx.gps.AbstractPosterior, float]:
            """Inner loop over multiple datasets y for a given
            Node with a given posterior model."""
            jax.clear_caches()
            total_max_log_likelihood = 0.0
            for y, stddev in zip(self.y.T, self.obs_stddev):
                posterior = set_obs_stddev(posterior, stddev)  # update stddev
                posterior, max_log_likelihood = fit_function(
                    posterior, self.X, y.reshape(-1, 1)
                )
                total_max_log_likelihood += max_log_likelihood
            return posterior, total_max_log_likelihood

        tqdm_object = tqdm(
            [deepcopy(node.posterior) for node in layer],
            desc=f"Fitting Layer {current_depth +1}",
            disable=False if self.verbosity == 1 else True,
        )

        if self.num_threads <= 1:
            fits = []
            for posterior in tqdm_object:
                if self.verbosity >= 2:
                    print(f"Current kernel: {_describe_kernel(posterior)}")
                fits.append(inner_loop(posterior))
        else:
            posteriors = [node.posterior for node in layer]
            with tqdm_joblib(tqdm_object):
                fits = Parallel(n_jobs=self.num_threads)(
                    delayed(inner_loop)(posterior) for posterior in posteriors
                )

        for node, fit in zip(layer, fits):
            node.update(*fit)  # type: ignore

    def select_top_nodes(
        self,
        layer: list[Node],
        n_leafs: int,
    ) -> list[Node]:
        """Select top nodes of current layer, based on
        their AIC/BIC value.

        Parameters
        ----------
        layer : list[Node]
            List of nodes in the current layer.
        n_leafs : int
            Number top nodes to keep.

        Returns
        -------
        list[Node]
            Sorted list of top nodes.
        """
        # sort with id in tuple, so that no errors is thrown if multiple
        # AIC/BIC are the same
        top_nodes = sorted(
            layer, key=lambda node: (self.get_criterion(node), id(node))
        )[:n_leafs]
        # return first n_leafs nodes
        return top_nodes

    def expand_layer(
        self,
        layer: list[Node],
    ) -> list[Node]:
        """Expand nodes in the current layer.

        Parameters
        ----------
        layer : list[Node]
            Layer of current (top) nodes.

        Returns
        -------
        list[Node]
            List of nodes in next layer.
        """
        next_layer = []
        for node in layer:
            self.expand_node(node)
            next_layer.extend(node.children)
        return next_layer

    def search(
        self,
        depth: int = 10,
        n_leafs: int = 3,
        patience: int = 1,
    ) -> Node:
        """Search for the best kernel fitting the data
        by performing a greedy search through possible kernel
        combinations.
        Start with simple kernel, which gets progressively more
        complex by adding or multiplying new kernels from kernel
        library. Kernels are evaluated by calculating their AIC/BIC
        after being fit to data.
        If the provided data y contains multiple datasets, the kernel
        structure which fits all data the best is returned. (In that case
        parameter and obs_stddev need to adjusted for each individual
        dataset afterwards.)

        Parameters
        ----------
        depth : int, optional
            The number of layers of the search tree. Deeper layers
            correspond to more complex kernels, by default 10
        n_leafs : int, optional
            The number of kernels to keep and expand at each layer. Top
            kernels are chosen based on AIC/BIC, by default 3
        patience : int, optional
            Number of layers to calculate without improving before early
            stopping, by default 1

        Returns
        -------
        gpx.gps.AbstractPosterior
            The node containing the best fitted kernel.

        """
        if self.clear_caches:
            jax.clear_caches()

        layer = self.root
        all_nodes = []

        best_model = None
        current_depth = 0
        criterion_threshold = jnp.inf
        patience_counter = 0
        for current_depth in range(depth):
            # fit and compute AIC at current layer
            self.compute_layer(layer, current_depth)
            if current_depth == 0:
                best_model = sorted(
                    layer, key=lambda node: (self.get_criterion(node), id(node))
                )[-1]
            all_nodes.extend(layer)

            # calculate and sort criterion
            current_crits = sorted([self.get_criterion(node) for node in layer])
            if self.verbosity >= 1:
                print(
                    f"Layer {current_depth+1} || "
                    f"Current top {self.criterion.upper()}s: "
                    f"{current_crits[:n_leafs]}"
                )

            # select best mdeols
            layer = self.select_top_nodes(layer, n_leafs)

            # Early stopping if no more improvements are found
            if current_crits[0] >= criterion_threshold:
                if patience_counter >= patience:
                    if self.verbosity >= 1:
                        print("No more improvements found! Terminating early..\n")
                    break
                patience_counter += 1
            else:
                best_model = layer[0]
                criterion_threshold = current_crits[0]  # min AIC/BIC of current layer
                patience_counter = 0

            # expand tree and search for new top noded in next layer down
            layer = self.expand_layer(layer)

        if best_model is None:
            raise ValueError("Loop did not run. Is depth>0?")

        if self.verbosity >= 1:
            print(f"Terminated on layer: {current_depth+1}.")
            print(f"Final log likelihood: {best_model.max_log_likelihood}")
            print(best_model.max_log_likelihood)
            print(f"Final number of model parameter: {best_model.n_parameter}")

        # save all evaluated nodes, sorted by AIC/BIC
        all_nodes = sorted(
            all_nodes, key=lambda node: (self.get_criterion(node), id(node))
        )
        self.nodes = all_nodes
        return best_model
