import warnings
from copy import deepcopy
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from beartype.typing import Callable
from gpjax.base import meta_map
from gpjax.fit import FailedScipyFitError
from jax import config, jit, tree_map
from jax.flatten_util import ravel_pytree
from jax.stages import Wrapped
from jax.tree_util import tree_leaves, tree_structure
from jaxtyping import Array, install_import_hook
from numpy.typing import NDArray
from tqdm import tqdm

from .kernels import (
    CombinationKernel,
    ProductKernel,
    SumKernel,
    flatten_kernels,
    get_kernel_info,
)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

# Enable Float64 for more stable matrix inversions.
config.update("jax_enable_x64", True)
# Throw error if any of the jax operations evaluate to nan.
# config.update("jax_debug_nans", True)

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

        self._update_node(
            posterior,
            max_log_likelihood,
        )

    def describe_kernel(self) -> str:
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
        return describe_kernel(self.posterior.prior.kernel)

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

    def update_trainables(
        self,
        parameter: tuple | list | Array | NDArray,
        unconstrain: bool = False,
    ) -> None:
        """Update posterior with new trainable parameter.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
                The gpjax posterior model.
        parameter : tuple | list | Array | NDArray
            The list of new parameter. Must be of same length
            as the number of trainable parameter
        unconstrain : bool, optional
            If True, model parameter are pushed trough bijector first
            making their support encompass the entire real line. The given
            parameter list must then be in this unconstrained space, by
            default False

        """
        self.posterior: gpx.gps.AbstractPosterior = set_trainables(
            self.posterior, parameter, unconstrain
        )

    def _update_node(
        self,
        posterior: gpx.gps.AbstractPosterior,
        max_log_likelihood: Optional[float] = None,
    ) -> None:
        """Update the new by setting new posterior,
        max_log_likelihood and n_posterior.
        Automatically calculates n_parameter from the model,
        and AIC if max_log_likelihood  is available.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
        max_log_likelihood : Optional[float], optional
             The log likelihood found after fittng, by default None
        parent : Optional[Node], optional
        """
        self.posterior = posterior

        self.n_parameter = sum(ravel_pytree(posterior.trainables())[0])

        if max_log_likelihood is not None:
            self.max_log_likelihood = max_log_likelihood
            self.aic = self.n_parameter * 2 - 2 * self.max_log_likelihood

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
        obs_stddev: float | Array = 1,
        fit_obs_stddev: bool = False,
        likelihood: Optional[gpx.likelihoods.AbstractLikelihood] = None,
        objective: Optional[gpx.objectives.AbstractObjective | Wrapped] = None,
        mean_function: Optional[gpx.mean_functions.AbstractMeanFunction] = None,
        root_kernel: Optional[gpx.kernels.AbstractKernel] = None,
        fitting_mode: str = "scipy",
        num_iters: int = 1000,
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
            Array of y data.
        obs_stddev : float | Array, optional
            The standard deviation of the y data. Is ignored if custom
            likelihood is given, by default 1
        fit_obs_stddev : bool, optional
            Wether to estimate the standard deviation during fitting process. Is
            ignored if custom likelihood is given, by default False
        likelihood : Optional[gpx.likelihoods.AbstractLikelihood], optional
            Function that calculates the likelihood of the model. By default None,
            which defaults to the Gaussian likelihood with standard deviation
            given by obs_stddev.
        objective : Optional[gpx.objectives.AbstractObjective  |  Wrapped], optional
            The objective function used to evalute the quality of a fit. By default
            None, which defaults to the jit-compiled marginal log likelihood.
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

        # set defaults
        if likelihood is None:
            likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=len(X), obs_stddev=obs_stddev
            )
            likelihood = likelihood.replace_trainable(
                obs_stddev=fit_obs_stddev  # type: ignore
            )  # type: ignore
        if objective is None:
            objective = jit(gpx.objectives.ConjugateMLL(negative=True))
        if mean_function is None:
            mean_function = gpx.mean_functions.Zero()
            # if not jnp.isclose(jnp.mean(y), 0):
            #    raise ValueError(
            #        "If no mean function is given, data must be centered at 0. "
            #        "Please center data first using y - mean(y)."
            #    )

        self.likelihood = likelihood
        self.objective = objective

        self.data = gpx.Dataset(
            X=X.reshape(-1, 1) if X.ndim == 1 else X,
            y=y.reshape(-1, 1) if y.ndim == 1 else y,
        )
        self.kernel_library = kernel_library
        self.nodes: list[Node] = []

        self.fitting_mode = fitting_mode
        self.num_iters = num_iters

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

    def fit(
        self, posterior: gpx.gps.AbstractPosterior
    ) -> tuple[gpx.gps.AbstractPosterior, float]:
        """Fit the hyperparameter of a posterior. Can be done using
        scipy's 'minimize' function using the 'adam' optimiser from
        optax.

        Parameters
        ----------
        posterior : gpx.gps.AbstractPosterior
            Posterior model object containing the hyperparameter.

        Returns
        -------
        tuple[gpx.gps.AbstractPosterior, float]
            Returns the posterior with optimised hyperparameter and
            log likelihood found at maximum.

        Raises
        ------
        ValueError
            Thrown if optimiser mode is unknown.
        """
        if self.fitting_mode == "scipy":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    optimized_posterior, history = gpx.fit_scipy(
                        model=posterior,
                        objective=self.objective,  # type: ignore
                        train_data=self.data,
                        max_iters=self.num_iters,
                        verbose=self.verbosity >= 2,
                    )
                except FailedScipyFitError:
                    optimized_posterior, history = posterior, [jnp.inf]

        elif self.fitting_mode == "adam":
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
                train_data=self.data,
                optim=optim,
                key=key,
                num_iters=self.num_iters,
                verbose=self.verbosity >= 2,
            )
        else:
            raise ValueError("'fitting_mode' must be 'scipy' or 'adam'.")

        max_log_likelihood = -float(history[-1])
        if not jnp.isfinite(max_log_likelihood):
            max_log_likelihood = -jnp.inf

        return optimized_posterior, max_log_likelihood

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
                    # have one multiplicative constant per term
                    try:
                        new_kernel = new_kernel.replace_trainable(
                            variance=False  # type: ignore
                        )
                    except ValueError:
                        pass

                    # multiply each term inidivually, if current kernel
                    # already is a sum
                    if (
                        isinstance(kernel, CombinationKernel)
                        and kernel.operator == jnp.sum
                    ):
                        composite_kernels = []
                        terms = kernel.kernels
                        assert terms
                        for i in range(len(terms)):
                            new_terms = deepcopy(terms)
                            new_terms[i] = kernel_operation(
                                kernels=[new_terms[i], new_kernel]
                            )
                            composite_kernels.append(SumKernel(kernels=new_terms))

                    else:
                        composite_kernels = [
                            kernel_operation(kernels=[kernel, new_kernel])
                        ]  # type: ignore
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

        Parameters
        ----------
        layer : list[Node]
            List of nodes in the current layer.
        current_depth : int
            Integer depth of current layer, used for
            tracking.
        """
        for node in tqdm(
            layer,
            desc=f"Fitting Layer {current_depth +1}",
            disable=False if self.verbosity == 1 else True,
        ):
            if self.verbosity >= 2:
                print(f"Current kernel: {describe_kernel(node.posterior.prior.kernel)}")
            node._update_node(*self.fit(node.posterior))

    def select_top_nodes(
        self,
        layer: list[Node],
        n_leafs: int,
    ) -> list[Node]:
        """Select top nodes of current layer, based on
        their AIC value.

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
        # AIC are the same
        top_nodes = sorted(layer, key=lambda node: (node.aic, id(node)))[:n_leafs]
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
    ) -> gpx.gps.AbstractPosterior:
        """Search for the best kernel fitting the data
        by performing a greedy search through possible kernel
        combinations.
        Start with simple kernel, which gets progressively more
        complex by adding or multiplying new kernels from kernel
        library. Kernels are evaluated by calculating their AIC
        after being fit to data.

        Parameters
        ----------
        depth : int, optional
            The number of layers of the search tree. Deeper layers
            correspond to more complex kernels, by default 10
        n_leafs : int, optional
            The number of kernels to keep and expand at each layer. Top
            kernels are chosen based on AIC, by default 3
        patience : int, optional
            Number of layers to calculate without improving before early
            stopping, by default 1

        Returns
        -------
        gpx.gps.AbstractPosterior
            The fitted gpjax posterior object
            for the best kernel.
        list[Node]
            A list of all nodes that were computed in the tree (including)
            their posteriors, sorted by their AIC value. Only returned
            if return_full is True

        """
        if self.clear_caches:
            jax.clear_caches()

        layer = self.root
        all_nodes = []

        best_model = None
        current_depth = 0
        aic_threshold = jnp.inf
        patience_counter = 0
        for current_depth in range(depth):
            # fit and compute AIC at current layer
            self.compute_layer(layer, current_depth)
            if current_depth == 0:
                best_model = sorted(layer, key=lambda node: (node.aic, id(node)))[-1]
            all_nodes.extend(layer)

            # calculate and sort AICs
            current_aics = sorted([float(node.aic) for node in layer])
            if self.verbosity >= 1:
                print(
                    f"Layer {current_depth+1} || Current top AICs: "
                    f"{current_aics[:n_leafs]}"
                )

            # select best mdeols
            layer = self.select_top_nodes(layer, n_leafs)

            # Early stopping if no more improvements are found
            if current_aics[0] > aic_threshold:
                if patience_counter >= patience:
                    if self.verbosity >= 1:
                        print("No more improvements found! Terminating early..\n")
                    break
                patience_counter += 1
            else:
                best_model = layer[0]
                aic_threshold = current_aics[0]  # min Aic of current layer
                patience_counter = 0

            # expand tree and search for new top noded in next layer down
            layer = self.expand_layer(layer)

        if best_model is None:
            raise ValueError("Loop did not run. Is depth>0?")

        if self.verbosity >= 1:
            print(f"Terminated on layer: {current_depth+1}.")
            print(f"Final log likelihood: {best_model.max_log_likelihood}")
            print(f"Final number of model parameter: {best_model.n_parameter}")

        # save all evaluated nodes, sorted by AIC
        all_nodes = sorted(all_nodes, key=lambda node: (node.aic, id(node)))
        self.nodes = all_nodes
        return best_model.posterior


def get_trainables(
    posterior: gpx.gps.AbstractPosterior,
    unconstrain: bool = False,
) -> Array:
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
    if unconstrain:
        posterior = posterior.unconstrain()

    all_params = ravel_pytree(posterior)[0]
    trainable_mask = ravel_pytree(posterior.trainables())[0]
    return all_params[trainable_mask]


def set_trainables(
    posterior: gpx.gps.AbstractPosterior,
    parameter: tuple | list | Array | NDArray,
    unconstrain: bool = False,
) -> gpx.gps.AbstractPosterior:
    """Returns posterior with updated trainable parameter.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
    parameter : tuple | list | Array | NDArray
        The list of new parameter. Must be of same length
        as the number of trainable parameter
    unconstrain : bool, optional
        If True, model parameter are pushed trough bijector first
        making their support encompass the entire real line. The given
        parameter list must then be in this unconstrained space, by
        default False

    Returns
    -------
    gpx.gps.AbstractPosterior
        The posterior with updated parameter.
    """

    def create_parameter_updater() -> Callable:
        # Check if the number of trainable parameters matches the length
        # of the parameter list
        num_trainable_params = sum(ravel_pytree(posterior.trainables())[0])
        if len(parameter) != num_trainable_params:
            raise ValueError(
                f"The length of the parameter list ({len(parameter)}) must "
                "match the number of trainable parameters "
                f"({num_trainable_params}) in the model."
            )

        # create iterable iterates through values in parameter list
        # everytime its called
        param_iterator = iter(parameter)

        # filter leaves, and assign new parameter from parameter list
        # if trainable is found
        def update_trainable(meta_leaf: tuple[dict, Array]) -> Array:
            meta, leaf = meta_leaf
            if meta.get("trainable", False):
                try:
                    return jnp.array(next(param_iterator))
                except StopIteration:
                    raise IndexError(
                        "Found more parameter in paramter list than "
                        "trainable parameters in model."
                    )
            else:
                return leaf

        return update_trainable

    updater = create_parameter_updater()

    if unconstrain:
        posterior = posterior.unconstrain()
    return meta_map(updater, posterior)  # type: ignore


@jit
def jit_set_trainables(
    posterior: gpx.gps.AbstractPosterior, parameter: Array, trainable_idx: Array
) -> gpx.gps.AbstractPosterior:
    """A jit-compatible version of set_trainables. For this, the indices of the
    trainable parameter must be given explicitly. There's no initial check of
    lengths, so make sure the length of the parameter array is the same as the
    number of trainable parameter.

    Parameters
    ----------
    posterior : gpx.gps.AbstractPosterior
            The gpjax posterior model.
    parameter : Array
        The array of new parameter. Must be of same length
        as the number of trainable parameter, and a jnp array.
    trainable_idx : Array
        A jnp array containing the indices of the trainable parameter.

    Returns
    -------
    gpx.gps.AbstractPosterior
        The posterior with updated parameter.
    """
    old_parameter = jnp.array(tree_leaves(posterior))
    updated_parameter = old_parameter.at[trainable_idx].set(parameter)

    new_posterior = tree_structure(posterior).unflatten(updated_parameter)
    return new_posterior


def describe_kernel(
    kernel: gpx.kernels.AbstractKernel
    | gpx.gps.AbstractPosterior
    | gpx.gps.AbstractPrior,
) -> str:
    """
    Generate string description of current kernel. Works with nested
    CombinationKernels in the kernel tree, but only those created
    explicity be the kernel search and its particular structure.

    Parameters
    ----------
    kernel :  gpx.kernels.AbstractKernel
            | gpx.gps.AbstractPosterior
            | gpx.gps.AbstractPrior
        The kernel to be described. Can also pass posterior or
        prior object, in which case the associated kernel is
        described.

    Returns
    -------
    str
        String description of kernel.
    """
    if isinstance(kernel, gpx.gps.AbstractPosterior):
        kernel = kernel.prior.kernel
    elif isinstance(kernel, gpx.gps.AbstractPrior):
        kernel = kernel.kernel
    elif isinstance(kernel, gpx.kernels.AbstractKernel):
        pass
    else:
        raise ValueError("'kernel' must be kernel, prior or posterior instance.")
    assert isinstance(kernel, gpx.kernels.AbstractKernel)

    def get_kernel_name(k: gpx.kernels.AbstractKernel) -> str:
        if isinstance(k, CombinationKernel):
            assert k.kernels
            sub_names = [describe_kernel(sub_k) for sub_k in k.kernels]
            joined_sub_names = (
                " • ".join(sub_names)
                if k.operator == jnp.prod
                else " + ".join(sub_names)
            )

            # Wrap in parentheses only if it's not the top-level kernel
            return f"{joined_sub_names}"
        else:
            return "Const" if hasattr(k, "constant") else f"{k.name}"

    return get_kernel_name(kernel)


def print_kernel_summary(
    kernel: gpx.kernels.AbstractKernel
    | gpx.gps.AbstractPosterior
    | gpx.gps.AbstractPrior,
) -> None:
    """
    Prints description of kernel as determined by kernel search.
    Works with nested CombinationKernels in the kernel tree, but only
    those created explicity be the kernel search and its particular
    structure.

    Parameters
    ----------
    kernel :  gpx.kernels.AbstractKernel
            | gpx.gps.AbstractPosterior
            | gpx.gps.AbstractPrior
        The kernel to be described. Can also pass posterior or
        prior object, in which case the associated kernel is
        described.
    """
    if isinstance(kernel, gpx.gps.AbstractPosterior):
        kernel = kernel.prior.kernel
    elif isinstance(kernel, gpx.gps.AbstractPrior):
        kernel = kernel.kernel
    elif isinstance(kernel, gpx.kernels.AbstractKernel):
        pass
    else:
        raise ValueError("'kernel' must be kernel, prior or posterior instance.")
    assert isinstance(kernel, gpx.kernels.AbstractKernel)

    if hasattr(kernel, "kernels"):
        kernels = flatten_kernels(kernel.kernels)  # type: ignore
    else:
        kernels = [kernel]

    # Header
    print("Kernel Summary\n")
    print("=" * 80)

    # Kernel description
    kernel_description = describe_kernel(kernel)
    print(f"Kernel Structure: {kernel_description}\n")
    print("-" * 80)

    # Column headers
    print(f"{'Kernel':<20} {'Property':<20} {'Value':<20} {'Trainable':<10}")
    print("-" * 80)

    # Individual kernel properties
    for k in kernels:
        kernel_info = get_kernel_info(k)
        if kernel_info:
            print(f"{k.name:<20}", end="")  # Kernel name in the first column
            for idx, (name, value, trainable) in enumerate(kernel_info):
                if idx > 0:
                    print(" " * 20, end="")  # Align subsequent rows
                print(f"{name:<20} {str(value):<20} {str(trainable):<10}")
                if idx < len(kernel_info) - 1:
                    print()
            print("-" * 80)
