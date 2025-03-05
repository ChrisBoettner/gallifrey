from functools import partial

import beartype.typing as tp
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import nnx
from flax.core import FrozenDict
from jax import jit, lax, vmap
from jaxtyping import Bool, Float, Int, PRNGKeyArray

from gallifrey.kernels.atoms import AbstractAtom, AbstractOperator
from gallifrey.kernels.library import KernelLibrary
from gallifrey.kernels.tree import TreeKernel
from gallifrey.utils.tree_helper import (
    calculate_max_depth,
    calculate_max_leaves,
    calculate_max_nodes,
    calculate_max_stack_size,
    get_child_idx,
    get_depth,
)
from gallifrey.utils.typing import ScalarBool, ScalarFloat, ScalarInt


class KernelPrior:
    """
    A prior distribution over kernel structures and parameters.

    Attributes
    ----------
    kernel_library : KernelLibrary
        An instance of the KernelLibrary class, containing
        the kernel classes and operators, and transformation functions.
    kernel_structure_prior : TreeStructurePrior
        A prior distribution over kernel structures.
    parameter_prior : ParameterPrior
        A prior distribution over kernel parameters.
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.
    num_datapoints : ScalarInt
        The number of data points in the dataset.
    max_kernel_parameter : int
        The maximum number of kernel parameters (max_atom_parameters
        * max_leaves).

    graphdef : nnx.GraphDef
        The graph definition of the kernel. Used together with the
        kernel state to create a TreeKernel object.

    kernels : tp.List[tp.Type[TreeKernel]]
        A list of kernel classes. (inherited from the kernel_library,
        see gallifrey.kernels.library.KernelLibrary)
    operators : tp.List[tp.Callable]
        A list of operators. (inherited from the kernel_library,
        see gallifrey.kernels.library.KernelLibrary)
    is_operator : Bool[jnp.ndarray, " D"]
        An array that indicates whether each kernel in the library is an operator.
        (inherited from the kernel_library, see gallifrey.kernels.library.KernelLibrary)
    probs :  Float[jnp.ndarray, " D"]
        The probabilities of sampling each kernel (or operator) in the library.
        (inherited from the kernel_structure_prior,
          see gallifrey.kernels.prior.TreeStructurePrior)
    prior_transforms : FrozenDict[str, tfb.Bijector]
        A (frozen) dictionary containing bijectors for transforming the
        distribution of the sampled parameters from a standard
        normal to the desired prior distribution. (inherited from the kernel_library,
        see gallifrey.kernels.library.KernelLibrary)
    support_bijectors : tuple[tfb.Bijector, ...]
        A tuple of bijectors for transforming the parameters from a constrained
        support space to an unconstrained space. Primarely used for the
        numerically stable optimization and sampling of the parameters (in a
        jit-compatible way).

    """

    def __init__(
        self,
        kernel_library: KernelLibrary,
        max_depth: int,
        num_datapoints: int,
        probs: tp.Optional[Float[jnp.ndarray, " D"]] = None,
    ) -> None:
        """
        Initialize the KernelPrior class.

        Parameters
        ----------
        kernel_library : KernelLibrary
            An instance of the KernelLibrary class, containing
            the kernel classes and operators, and transformation functions.
            The TreeStructurePrior and ParameterPrior classes are initialized
            using the kernel_library.
        max_depth : int
            The maximum depth of the nested kernel structure tree.
        num_datapoints : int
            The number of data points in the dataset.
        probs : tp.Optional[ Float[jnp.ndarray, " D"]], optional
            The probabilities of sampling each kernel (or operator) in the library.
            The array must have the same length as the the arrays in the kernel library.
            By default None, which will use a uniform distribution.
        """
        self.kernel_library = kernel_library

        self.kernel_structure_prior = TreeStructurePrior(
            self.kernel_library,
            max_depth,
            probs,
        )

        self.num_datapoints = num_datapoints

        self.parameter_prior = ParameterPrior(self.kernel_library, max_depth)

        self.graphdef = self._get_graphdef()

        self.max_kernel_parameter = (
            kernel_library.max_atom_parameters * calculate_max_leaves(self.max_depth)
        )

        self.support_bijectors = tuple(
            [
                self.kernel_library.support_transforms[tag]
                for tag in self.kernel_library.support_tag_mapping
            ]
        )

    def sample(self, key: PRNGKeyArray) -> nnx.State:
        """
        Sample a kernel structure and its parameters from the prior distribution.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.

        Returns
        -------
        nnx.State
        The state of the sampled kernel, combine with the graphdef
        to create a TreeKernel object.

        """
        tree_key, parameter_key = jr.split(key)
        kernel_structure = self.kernel_structure_prior.sample(
            tree_key,
            self.kernel_structure_prior.max_depth,
        )

        kernel = TreeKernel(
            kernel_structure,
            self.kernel_library,
            self.max_depth,
            self.num_datapoints,
        )

        _, state = nnx.split(kernel)

        kernel_state, _ = self.parameter_prior.sample(
            parameter_key,
            nnx.State(state),
        )
        return kernel_state

    def sample_kernel(self, key: PRNGKeyArray) -> TreeKernel:
        """
        Sample a kernel state using self.sample and merge
        it with the graphdef and static_state to create a
        TreeKernel.

        This function is convenient for sampling a kernel
        directly, but not vmap or jittable (at least not
        using the jax commands, potentially using the
        nnx ones.)

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.

        Returns
        -------
        TreeKernel
            The sampled kernel.
        """

        kernel_state = self.sample(key)
        return nnx.merge(self.graphdef, kernel_state)

    def reconstruct_kernel(
        self,
        kernel_state: nnx.State,
    ) -> TreeKernel:
        """
        Create new TreeKernel from kernel state using the graphdef,
        and reset some kernel attributes. In principle,
        this is unnecessary since the values are already
        set in the kernel state, but we need to make it
        explicit for the jit compilation.

        Parameters
        ----------
        kernel_state : nnx.State
            The kernel state to be used to create the kernel.

        Returns
        -------
        TreeKernel
            The TreeKernel instance.
        """

        kernel = nnx.merge(self.graphdef, kernel_state)

        max_depth = self.max_depth
        max_nodes = calculate_max_nodes(max_depth)
        max_leaves = calculate_max_leaves(max_depth)
        max_stack = calculate_max_stack_size(max_depth)
        kernel.max_depth = kernel.max_depth.replace(max_depth)
        kernel.max_nodes = kernel.max_nodes.replace(max_nodes)
        kernel.max_leaves = kernel.max_leaves.replace(max_leaves)
        kernel.max_stack = kernel.max_stack.replace(max_stack)
        kernel.num_atoms = kernel.num_atoms.replace(self.kernel_library.num_atoms)
        kernel.num_datapoints = kernel.num_datapoints.replace(self.num_datapoints)
        return kernel

    def _get_graphdef(
        self,
    ) -> nnx.GraphDef:
        """
        Create a random kernel and return the graphdef.

        This function is used to create a graphdef for the
        kernel (which is set as attribute in the __init__).
        The graphdef should be the same for all possible
        kernels, so we can reuse it whenever we need to
        create a KernelTree object from a kernel state.

        Returns
        -------
        nnx.GraphDef
            The graph definition of the kernel.
        """

        key = jr.PRNGKey(42)

        kernel_structure = self.kernel_structure_prior.sample(
            key,
            self.kernel_structure_prior.max_depth,
        )

        kernel = TreeKernel(
            kernel_structure,
            self.kernel_library,
            self.max_depth,
            self.num_datapoints,
        )

        graphdef, _ = nnx.split(kernel)
        return graphdef

    @property
    def atoms(self) -> tp.List[AbstractAtom]:
        return self.kernel_library.atoms

    @property
    def operators(self) -> tp.List[AbstractOperator]:
        return self.kernel_library.operators

    @property
    def is_operator(self) -> Bool[jnp.ndarray, " D"]:
        return self.kernel_library.is_operator

    @property
    def probs(self) -> Float[jnp.ndarray, " D"]:
        return self.kernel_structure_prior.probs

    @property
    def max_depth(self) -> int:
        return self.kernel_structure_prior.max_depth

    @property
    def prior_transforms(self) -> FrozenDict[str, tfb.Bijector]:
        return self.kernel_library.prior_transforms


@partial(
    jit,
    static_argnames=(
        "max_leaves",
        "max_atom_parameters",
        "forward_bijectors",
    ),
)
def sample_parameters(
    key: PRNGKeyArray,
    state: nnx.State,
    considered_nodes: Int[jnp.ndarray, " D"],
    num_parameter_array: Int[jnp.ndarray, " D"],
    max_leaves: int,
    max_atom_parameters: int,
    support_mapping_array: Int[jnp.ndarray, "M N"],
    forward_bijectors: tuple[tp.Callable, ...],
) -> tuple[nnx.State, ScalarFloat]:
    """
    A function to sample new parameters for a state, and apply them to
    the kernel. Also returns the log probability of the sampled parameters.

    The parameters are sampled from a standard normal. Whether a
    parameter is sampled or not is determined by the tree expression
    and the leaf level map (See 'gallifrey.kernels.tree' for more
    information).
    The parameter are transformed to the desired prior distribution
    using the support_mapping_array and the forward_bijectors.

    We include an input 'considered_nodes' to only sample a specific
    subset of nodes. This is useful for structure moves, where only
    a subset of the parameters are changed. (Since arrays need to
    be of fixed shape, consider_nodes can be padded with -1 or
    any value that's not a valid leaf index.)

    Inactive parameters are set to -1.

    Parameters
    ----------
    key : PRNGKeyArray
        Random key for sampling.
    state : nnx.State
        The original kernel state to be filled with new parameters.
        The state must contain the following attributes:
        - tree_expression: The tree expression that describes the kernel structure.
        - leaf_level_map: A array that maps the index of the leaf parameter array
    considered_nodes : Int[jnp.ndarray, " D"]
        This array needs to contain all the indices that are
        supposed to be sampled. For first time sampling, that
        should be all indices (or at least all leaves). For
        structure moves (e.g. subtree-replace move), it should
        contain the indices that have been changed from the
        previous tree.
    num_parameter_array : Int[jnp.ndarray, " D"]
        An array that contains the number of parameters for each
        atom in the kernel structure. Needs to be in same order
        as the atom library.
    max_leaves : int
        The maximum number of leaves in the tree.
    max_atom_parameters : int
        The maximum number of parameters any atom will take.
    support_mapping_array : Int[jnp.ndarray, "M N"]
        An array that maps the support of the parameters to the
        corresponding bijector index.
    forward_bijectors : tuple[tp.Callable, ...]
        A tuple of bijectors to transform the sampled parameters from
        a standard normal to the desired prior distribution.

    Returns
    -------
    nnx.State
        The state with the sampled parameters.
    ScalarFloat
        The (total) log probability of the sampled parameters, sum
        of all individual log probabilities.

    """

    tree_expression: jnp.ndarray = state.tree_expression.value  # type: ignore
    leaf_level_map: jnp.ndarray = state.leaf_level_map.value  # type: ignore

    def process_params(
        state: tuple[Float[jnp.ndarray, "M N"], Float[jnp.ndarray, ""], PRNGKeyArray],
        indices: Int[jnp.ndarray, "2"],
    ) -> tuple[
        tuple[Float[jnp.ndarray, "M N"], Float[jnp.ndarray, ""], PRNGKeyArray],
        Int[jnp.ndarray, "2"],
    ]:
        """Process parameter based on if it's active and considered for sampling."""

        # unpack state and indices
        kernel_parameter, log_probability, key = state
        leaf_idx, parameter_idx = indices

        # check whether the parameter is active
        node_value = tree_expression[leaf_level_map[leaf_idx]]
        atom_num_parameters = num_parameter_array[node_value]

        is_active_node = leaf_level_map[leaf_idx] >= 0
        is_active_param = parameter_idx < atom_num_parameters
        is_active = is_active_node & is_active_param

        # check whether the parameter node is part of the considered nodes
        considered_for_sampling = jnp.isin(
            leaf_level_map[leaf_idx],
            considered_nodes,
        )

        def process_active_and_considered(indices_and_key: tuple) -> tuple:
            """If active and considered for sampling, sample and transform the
            parameter, and calculate the log probability."""
            leaf_idx, parameter_idx, key = indices_and_key

            # sample parameter
            key, subkey = jr.split(key)
            sampled_param = jr.normal(subkey)

            # calculate log probability (standard normal)
            log_prob_param = -0.5 * (sampled_param**2 + jnp.log(2 * jnp.pi))

            # transform parameter to desired prior distribution
            transformed_param = lax.switch(
                support_mapping_array[node_value, parameter_idx],
                forward_bijectors,
                sampled_param,
            )
            return transformed_param, log_prob_param, key

        def process_active_not_considered(indices_and_key: tuple) -> tuple:
            """If active but not considered, return the current parameter."""
            leaf_idx, parameter_idx, key = indices_and_key
            return kernel_parameter[leaf_idx, parameter_idx], 0.0, key

        def process_inactive(indices_and_key: tuple) -> tuple:
            """If inactive, return -1."""
            _, _, key = indices_and_key
            return jnp.array(-1.0), 0.0, key  # inactive parameters are set to -1

        # process parameter, get new parameter and key
        sampled_param, log_prob_param, key = lax.cond(
            is_active,
            lambda _: lax.cond(
                considered_for_sampling,
                process_active_and_considered,
                process_active_not_considered,
                (leaf_idx, parameter_idx, key),
            ),
            process_inactive,
            (leaf_idx, parameter_idx, key),
        )

        # update kernel parameter
        kernel_parameter = kernel_parameter.at[leaf_idx, parameter_idx].set(
            sampled_param
        )

        # update the log probability
        log_probability = log_probability + log_prob_param

        return (kernel_parameter, log_probability, key), indices

    # get initial kernel parameters
    kernel_parameters: jnp.ndarray = state.parameters.value  # type: ignore

    # get all parameter index combinations
    parameter_indices = jnp.indices((max_leaves, max_atom_parameters)).reshape(2, -1).T

    # scan over all parameters and update kernel parameters
    final_state, _ = lax.scan(
        process_params,
        (kernel_parameters, jnp.array(0.0), key),
        parameter_indices,
    )
    new_kernel_parameters = final_state[0]
    tot_log_prob = final_state[1]

    state["parameters"] = state["parameters"].replace(
        new_kernel_parameters,
    )  # type: ignore
    return state, tot_log_prob


@partial(
    jit,
    static_argnames=(
        "max_leaves",
        "max_atom_parameters",
        "inverse_bijectors",
    ),
)
def log_prob_parameters(
    state: nnx.State,
    considered_nodes: Int[jnp.ndarray, " D"],
    num_parameter_array: Int[jnp.ndarray, " D"],
    max_leaves: int,
    max_atom_parameters: int,
    support_mapping_array: Int[jnp.ndarray, "M N"],
    inverse_bijectors: tuple[tp.Callable, ...],
) -> ScalarFloat:
    """
    A function takes a state and calculates the log probability of the
    kernel parameters.

    It is assumed the parameters are sampled from a standard normal and
    then transformed to the desired prior distribution using the
    support_mapping_array and the forward_bijectors. In this function,
    we transform the parameters back to the standard normal and
    calculate the log probability.

    We include an input 'considered_nodes' to only sample a specific
    subset of nodes. This is useful for structure moves, where only
    a subset of the parameters are changed. (Since arrays need to
    be of fixed shape, consider_nodes can be padded with -1 or
    any value that's not a valid leaf index.)

    Parameters
    ----------
    state : nnx.State
        The original kernel state to be filled with new parameters.
        The state must contain the following attributes:
        - tree_expression: The tree expression that describes the kernel structure.
        - leaf_level_map: A array that maps the index of the leaf parameter array
    considered_nodes : Int[jnp.ndarray, " D"]
        This array needs to contain all the indices that are
        supposed to be sampled. For first time sampling, that
        should be all indices (or at least all leaves). For
        structure moves (e.g. subtree-replace move), it should
        contain the indices that have been changed from the
        previous tree.
    num_parameter_array : Int[jnp.ndarray, " D"]
        An array that contains the number of parameters for each
        atom in the kernel structure. Needs to be in same order
        as the atom library.
    max_leaves : int
        The maximum number of leaves in the tree.
    max_atom_parameters : int
        The maximum number of parameters any atom will take.
    support_mapping_array : Int[jnp.ndarray, "M N"]
        An array that maps the support of the parameters to the
        corresponding bijector index.
    inverse_bijectors : tuple[tp.Callable, ...]
        A tuple of bijectors to transform the parameters back to
        a standard normal.

    Returns
    -------
    ScalarFloat
        The (total) log probability of the parameters, sum
        of all individual log probabilities.

    """

    tree_expression: jnp.ndarray = state.tree_expression.value  # type: ignore
    leaf_level_map: jnp.ndarray = state.leaf_level_map.value  # type: ignore

    def process_params(
        state: tuple[Float[jnp.ndarray, "M N"], Float[jnp.ndarray, ""]],
        indices: Int[jnp.ndarray, "2"],
    ) -> tuple[
        tuple[Float[jnp.ndarray, "M N"], Float[jnp.ndarray, ""]],
        Int[jnp.ndarray, "2"],
    ]:
        """Process parameter based on if it's active and considered."""

        # unpack state and indices
        kernel_parameters, log_probability = state
        leaf_idx, parameter_idx = indices

        # check whether the parameter is active
        node_value = tree_expression[leaf_level_map[leaf_idx]]
        atom_num_parameters = num_parameter_array[node_value]

        is_active_node = leaf_level_map[leaf_idx] >= 0
        is_active_param = parameter_idx < atom_num_parameters
        is_active = is_active_node & is_active_param

        # check whether the parameter node is part of the considered nodes
        considered_for_sampling = jnp.isin(
            leaf_level_map[leaf_idx],
            considered_nodes,
        )

        def process_active_and_considered(indices: tuple) -> ScalarFloat:
            """If active and considered for sampling, transform parameter back
            to standard normal and calculate probability."""
            leaf_idx, parameter_idx = indices

            # select parameter and transform back to standard normal
            param = kernel_parameters[leaf_idx, parameter_idx]

            transformed_param = lax.switch(
                support_mapping_array[node_value, parameter_idx],
                inverse_bijectors,
                param,
            )

            # calculate log probability (standard normal)
            log_prob_param = -0.5 * (transformed_param**2 + jnp.log(2 * jnp.pi))

            return log_prob_param

        def process_others(indices: tuple) -> ScalarFloat:
            """If not active or not considered for sampling, don't
            consider for log probability."""
            return jnp.array(0.0)

        # process parameter, get probabilities
        log_prob_param = lax.cond(
            is_active & considered_for_sampling,
            process_active_and_considered,
            process_others,
            (leaf_idx, parameter_idx),
        )

        # update the log probability
        log_probability += log_prob_param

        return (kernel_parameters, log_probability), indices

    # get initial kernel parameters
    kernel_parameters: jnp.ndarray = state.parameters.value  # type: ignore

    # get all parameter index combinations
    parameter_indices = jnp.indices((max_leaves, max_atom_parameters)).reshape(2, -1).T

    # scan over all parameters and get probabilities
    parameters_and_total_log_prob, _ = lax.scan(
        process_params,
        (kernel_parameters, jnp.array(0.0)),
        parameter_indices,
    )

    return parameters_and_total_log_prob[1]


class ParameterPrior:
    """
    This class defines a prior distribution over kernel parameters.

    Attributes
    ----------
    num_parameter_array : Int[jnp.ndarray, " D"]
        An array that contains the number of parameters for each
        atom in the kernel structure. Needs to be in same order
        as the atom library.
    max_atom_parameters : int
        The maximum number of parameters for an atom in the kernel structure.
    max_leaves : int
        The maximum number of leaves in the tree.
    max_nodes : int
        The maximum number of nodes in the kernel structure.
    support_mapping_array : Int[jnp.ndarray, " D"]
        An array that maps the support of the parameters to the
        corresponding bijector index.
    forward_bijectors : tuple[tfb.Callable]
        A tuple of bijectors to transform the sampled parameters from
        a standard normal to the desired prior distribution.
    inverse_bijectors : tuple[tfb.Callable]
        A tuple of bijectors to transform the sampled parameters from
        the prior distribution back to a standard normal.

    """

    def __init__(
        self,
        kernel_library: KernelLibrary,
        max_depth: int,
    ):
        """
        Initialize the ParameterPrior class.

        Parameters
        ----------
        kernel_library : KernelLibrary
            An instance of the KernelLibrary class, containing
            the kernel classes and operators, and transformation functions.
            The prior_transforms dictionary is used to transform the
            sampled parameters to the desired prior distribution.
        max_depth : int
            The maximum depth of the kernel tree.

        """
        self.num_parameter_array = jnp.array(
            [atom.num_parameter for atom in kernel_library.library]
        )

        self.max_atom_parameters = kernel_library.max_atom_parameters
        self.max_leaves = int(calculate_max_leaves(max_depth))
        self.max_nodes = int(calculate_max_nodes(max_depth))

        self.support_mapping_array = kernel_library.support_mapping_array

        # create tuples of callables for forward and inverse bijectors
        self.forward_bijectors = tuple(
            [
                kernel_library.prior_transforms[tag].forward  # type: ignore
                for tag in kernel_library.support_tag_mapping
            ]
        )

        self.inverse_bijectors = tuple(
            [
                kernel_library.prior_transforms[tag].inverse  # type: ignore
                for tag in kernel_library.support_tag_mapping
            ]
        )

    def sample(
        self,
        key: PRNGKeyArray,
        state: nnx.State,
    ) -> tuple[nnx.State, ScalarFloat]:
        """
        Sample kernel parameter and assign it to the kernel. Also
        returns the log probability of the sampled parameters.


        The kernel parameter are sampled from a standard normal and
        transformed to follow their corresponding prior distributions
        defined by the parameter_transforms dictionary.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.
        state : nnx.State
            The original kernel state to be filled with new parameters.

        Returns
        -------
        nnx.State
            The state with the sampled parameters.
        ScalarFloat
            The log probability of the sampled parameters.

        """
        return self.sample_subset(
            key,
            state,
            jnp.arange(self.max_nodes),  # all nodes are considered
        )

    def sample_subset(
        self,
        key: PRNGKeyArray,
        state: nnx.State,
        considered_nodes: Int[jnp.ndarray, " D"],
    ) -> tuple[nnx.State, ScalarFloat]:
        """
        Sample kernel parameter and assign it to the kernel.
        Same as 'sample' method but with additional
        parameter 'considered_nodes', which can be used
        to only sample parameters for a subset of nodes.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.
        state : nnx.State
            The original kernel state to be filled with new parameters.
        considered_nodes : Int[jnp.ndarray, " D"]
            An array that contains the indices of the nodes that
            are supposed to be sampled. (Padded with -1 if
            necessary.)

        Returns
        -------
        nnx.State
            The state with the sampled parameters.
        ScalarFloat
            The log probability of the sampled parameters.
        """

        new_state, log_prob = sample_parameters(
            key,
            state,
            considered_nodes,
            self.num_parameter_array,
            self.max_leaves,
            self.max_atom_parameters,
            self.support_mapping_array,
            self.forward_bijectors,
        )
        return new_state, log_prob

    def log_prob(
        self,
        state: nnx.State,
    ) -> ScalarFloat:
        """
        Compute the log probability of the kernel parameters.

        Parameters
        ----------
        state : nnx.State
            The kernel state with the parameters.

        Returns
        -------
        ScalarFloat
            The log probability of the kernel parameters.
        """

        return self.log_prob_subset(
            state,
            jnp.arange(self.max_nodes),  # all nodes are considered
        )

    def log_prob_subset(
        self,
        state: nnx.State,
        considered_nodes: Int[jnp.ndarray, " D"],
    ) -> ScalarFloat:
        """
        Compute the log probability of the kernel parameters.
        Same as 'log_prob' method but with additional
        parameter 'considered_nodes', which can be used
        to only calculate the log probability for a subset of nodes.

        Parameters
        ----------
        state : nnx.State
            The kernel state with the parameters.
        considered_nodes : Int[jnp.ndarray, " D"]
            An array that contains the indices of the nodes that
            are supposed to be sampled. (Padded with -1 if
            necessary.)

        Returns
        -------
        ScalarFloat
            The log probability of the kernel parameters.
        """

        return log_prob_parameters(
            state,
            considered_nodes,
            self.num_parameter_array,
            self.max_leaves,
            self.max_atom_parameters,
            self.support_mapping_array,
            self.inverse_bijectors,
        )


class TreeStructurePrior(tfd.Distribution):
    """
    A prior distribution over kernel structures.

    The TreeStructurePrior is a distribution over kernel structures. The kernel
    structure is represented as a tree, where each leaf corresponds to a kernel. The
    tree is constructed by sampling from a library of kernel classes and operators.
    The operators are used to construct nested kernel structures.

    Attributes
    ----------
    library : tp.List[tp.Type[AbstractKernel] | tp.Callable]
        A list of kernel classes and operators, as defined in the KernelLibrary class.
        See gallifrey.kernels.library.KernelLibrary for more details.
    is_operator : tp.List[bool]
        A list of booleans indicating whether each element in the library is an
        operator.
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.
    probs : jnp.ndarray
        The probabilities of sampling each kernel (or operator) in the library.
        See kernels.library.KernelLibrary and gallifrey.config.GPConfig for more
        details.

    """

    def __init__(
        self,
        kernel_library: KernelLibrary,
        max_depth: int,
        probs: tp.Optional[Float[jnp.ndarray, " D"]] = None,
        *,
        validate_args: tp.Optional[bool] = None,
    ):
        """
        Initialize the KernelPrior distribution.

        Parameters
        ----------
        kernel_library : KernelLibrary
            An instance of the KernelLibrary class, containing the kernel classes and
            operators. (See gallifrey.kernels.library.KernelLibrary)
        max_depth : int
            The maximum depth of the nested kernel structure tree.
        probs :  Float[jnp.ndarray, " D"], optional
            The probabilities of sampling each kernel (or operator) in the library.
            The array must have the same length as the the arrays in the kernel library.
            By default None, which will use a uniform distribution.
        validate_args : tp.Optional[bool], optional
            Whether to validate input, by default None. NOT IMPLEMENTED.

        Raises
        ------
        ValueError
            If the probs are not one-dimensional.
        ValueError
            If the probs do not have the same length as the kernel_library
            along the last dimension.
        """
        # initialize the kernel functions and the probabilities
        self.library = kernel_library.library
        self.is_operator = kernel_library.is_operator
        self.max_depth = max_depth

        self.probs = probs if probs is not None else jnp.ones(len(self.library))
        if jnp.ndim(self.probs) != 1:
            raise ValueError("'probs' must be one-dimensional.")
        if not len(self.library) == len(self.probs):
            raise ValueError(
                "'probs' must have same length along the last "
                "dimension as 'kernel_library'. "
                f"Got 'probs' shape: {jnp.shape(self.probs)}, "
                f"'kernel_library' length: {len(self.library)}."
            )

    def sample_single(
        self,
        key: PRNGKeyArray,
        max_depth: tp.Optional[ScalarInt] = None,
        root_idx: ScalarInt = 0,
    ) -> Int[jnp.ndarray, " D"]:
        """
        Construct an abstract representation of a kernel structure by sampling from
        the kernel library.
        The kernels are sampled from the library according to the defined
        probabilities.
        The structure is represented as a one-dimensional array, according to
        the level-order traversal of the tree. This means:
        - The root node is at position 0.
        - The left child of a node at position i is at position 2*i + 1.
        - The right child of a node at position i is at position 2*i + 2.
        - Empty nodes are labeled -1.
        See gallifrey.kernels.tree.TreeKernel for more details and an example.

        Parameters
        ----------
        key : PRNGKeyArray
            Random key for sampling.
        max_depth : ScalarInt, optional
            The maximum depth of the nested kernel structure tree, by default None.
            If None, the maximum depth is set to the value of self.max_depth.
        root_idx : ScalarInt, optional
            The index of the root node in the tree, by default 0. Used
            for sub-trees.

        Returns
        -------
        Int[jnp.ndarray, " D"]
            An array that describes the kernel structure.
        """
        max_depth = max_depth if max_depth is not None else self.max_depth

        if max_depth < 0:
            raise ValueError("'max_depth' must be 0 or larger.")

        max_nodes = calculate_max_nodes(max_depth)

        # create sample array to be filled, this will be the output (empty
        # nodes are labeled -1)
        sample = jnp.full(max_nodes, -1)
        # create initial stack: empty except for the root node
        initial_stack = jnp.copy(sample).at[0].set(root_idx)

        pointer = 0  # initial position of the stack pointer

        initial_state = (key, sample, initial_stack, pointer)

        return _sample_single(
            initial_state,
            self.probs,
            self.is_operator,
            max_depth,
        )

    def log_prob_single(
        self,
        value: Int[jnp.ndarray, " D"],
        root_idx: ScalarInt = 0,
        path_to_hole: tp.Optional[Int[jnp.ndarray, " D"]] = None,
        hole_idx: tp.Optional[ScalarInt] = None,
    ) -> ScalarFloat:
        """
        Compute the log probability of a given kernel structure,
        as represented by the level-order tree array.
        The maximum depth of the tree is inferred from the length of the array.

        The function can also be used to calculate the log probability of a
        scaffold structure (used by the detach-attach move), by providing
        the path to the hole and the hole index.

        Parameters
        ----------
        value :  Int[jnp.ndarray, " D"]
            An array that describes the kernel structure.
        root_idx : ScalarInt, optional
            The index of the root node in the tree, by default 0.
        path_to_hole :  Int[jnp.ndarray, " D"], optional
            The path to the hole in the tree. A list of indices that
            describe the path from the root to the hole (level order
            indices), by default None (sets it to an empty array).
        hole_idx : ScalarInt, optional
            The index of the hole in the tree (level order index),
            by default None(sets it to -1 which should never be
            reached).

        Returns
        -------
        ScalarFloat
            The log probability of the given kernel structure or scaffold.
        """
        max_nodes = len(value)
        max_depth = calculate_max_depth(max_nodes)

        initial_log_p = 0.0
        inital_stack = jnp.full(max_nodes, -1).at[0].set(root_idx)
        pointer = 0
        initial_state = (initial_log_p, inital_stack, pointer)

        # set the path to the hole and the hole index, if not given
        path_to_hole = path_to_hole if path_to_hole is not None else jnp.array([])
        hole_idx = hole_idx if hole_idx is not None else -1

        return _log_prob_single(
            initial_state,
            value,
            self.probs,
            self.is_operator,
            max_depth,
            path_to_hole,
            hole_idx,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_depth: tp.Optional[ScalarInt] = None,
        root_idx: ScalarInt = 0,
        sample_shape: tuple = (),
    ) -> Int[jnp.ndarray, "..."]:
        """
        Sample kernel structures. For details, see `sample_single`.

        Parameters
        ----------
        key : PRNGKey
            Random key for sampling.
        max_depth : ScalarInt
            The maximum depth of the nested kernel tree structure. If None,
            the maximum depth is set to the value of self.max_depth.
        root_idx : ScalarInt, optional
            The index of the root node in the tree, by default 0.
            Used for sub-trees.
        sample_shape : tuple, optional
            The sample shape for the distribution, by default ().

        Returns
        -------
        Int[jnp.ndarray, "..."]
            An array of samples describing kernel structures,
            of shape sample_shape + (max_nodes,).
        """
        max_depth = max_depth if max_depth is not None else self.max_depth

        if not sample_shape:
            return self.sample_single(key, max_depth, root_idx)

        max_nodes = calculate_max_nodes(max_depth)

        # flatten the sample_shape for vectorized sampling
        num_samples = jnp.prod(jnp.array(sample_shape))
        keys = jr.split(key, num=int(num_samples))

        samples = vmap(self.sample_single, in_axes=(0, None, None))(
            keys,
            max_depth,
            root_idx,
        )
        return samples.reshape(sample_shape + (max_nodes,))

    def log_prob(
        self,
        value: Int[jnp.ndarray, "..."],
        root_idx: ScalarInt = 0,
        path_to_hole: tp.Optional[Int[jnp.ndarray, "..."]] = None,
        hole_idx: tp.Optional[ScalarInt] = None,
    ) -> Float[jnp.ndarray, "..."]:
        """
        Compute the log probability of given kernel structures or scaffolds.
        (See log_prob_single for details.)

        Parameters
        ----------
        value :  Int[jnp.ndarray, "..."]
            Expression of kernel structure(s).
        root_idx : ScalarInt, optional
            The index of the root node in the tree, by default 0.
        path_to_hole :  Int[jnp.ndarray, "..."], optional
            Optional path to the hole in the tree, if calculating the log
            probability of a scaffold, by default None.
        hole_idx : ScalarInt, optional
            The index of the hole in the tree if calculating the log
            probability of a scaffold, by default None.

        Returns
        -------
         Float[jnp.ndarray, "..."]
            The log probability of the given kernel structures.
        """
        sample_shape = jnp.shape(value)[:-1]

        num_samples = jnp.prod(jnp.array(sample_shape))
        log_probs = vmap(self.log_prob_single, in_axes=(0, None, None, None))(
            value.reshape(int(num_samples), -1),
            root_idx,
            path_to_hole,
            hole_idx,
        )
        return jnp.asarray(log_probs).reshape(sample_shape)


@jit
def _sample_single(
    initial_state: tuple,
    probs: Float[jnp.ndarray, " D"],
    is_operator: Bool[jnp.ndarray, " D"],
    max_depth: ScalarInt,
) -> Int[jnp.ndarray, " D"]:
    """
    Construct an abstract representation of a kernel structure by sampling from
    the kernel library.
    The kernels are sampled from the library according to the defined
    probabilities.
    The structure is represented as a one-dimensional array, according to
    the level-order traversal of the tree. This means:
    - The root node is at position 0.
    - The left child of a node at position i is at position 2*i + 1.
    - The right child of a node at position i is at position 2*i + 2.
    - Empty nodes are labeled -1.
    See gallifrey.kernels.tree.TreeKernel for more details and an example.

    NOTE: This function is should be called via the sample_single method
    of the KernelPrior class.

    Parameters
    ----------
    initial_state : tuple
        Initial state for the while loop.
    probs :  Float[jnp.ndarray, " D"]
        The probabilities of sampling each kernel in the library.
    is_operator : Bool[jnp.ndarray, " D"]
        An array that indicates whether each kernel in the library is an operator.
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.

    Returns
    -------
     Int[jnp.ndarray, " D"]
        An array that describes the kernel structure.
    """

    def traverse_tree(
        state: tuple[PRNGKeyArray, jnp.ndarray, jnp.ndarray, ScalarInt]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ScalarInt]:
        """Traverse the tree and sample the kernel structure."""
        key, sample, stack, pointer = state

        # pop the top of the stack
        pos = stack[pointer]
        pointer = pointer - 1

        # compute the depth of the current node
        depth = get_depth(pos)

        # determine if operators are allowed at this depth
        is_at_max_depth = depth >= max_depth
        # mask the probabilities: if at max_depth, disable operator choices
        masked_probs = jnp.where(
            is_at_max_depth,
            jnp.where(is_operator, 0.0, probs),  # if at max depth, mask operators
            probs,  # else use full probs
        )

        # create subkey for sampling and next iteration
        key, new_key = jr.split(key)

        # sample choice from masked_probs
        choice = jr.categorical(
            new_key,
            logits=jnp.log(masked_probs),
        )
        sample_updated = sample.at[pos].set(choice)

        # determine if choice is an operator
        is_op = is_operator[choice]

        def push(
            stack_and_pointer: tuple[jnp.ndarray, ScalarInt]
        ) -> tuple[jnp.ndarray, ScalarInt]:
            """If the node is an operator, push the new
            child nodes onto the stack.
            """
            stack, pointer = stack_and_pointer

            left_child = get_child_idx(pos, "l")
            right_child = get_child_idx(pos, "r")

            stack_updated = (
                stack.at[pointer + 1].set(right_child).at[pointer + 2].set(left_child)
            )
            pointer_updated = pointer + 2
            return stack_updated, pointer_updated

        def dont_push(
            stack_and_pointer: tuple[jnp.ndarray, ScalarInt]
        ) -> tuple[jnp.ndarray, ScalarInt]:
            """If the node is a leaf node, do not modify the stack."""
            return stack_and_pointer

        # Conditionally sample the node or set it as empty based on parent operator
        stack_updated, pointer_updated = lax.cond(
            is_op,
            push,
            dont_push,
            operand=(stack, pointer),
        )

        return (
            new_key,
            sample_updated,
            stack_updated,
            pointer_updated,
        )

    def condition(
        state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ScalarInt]
    ) -> ScalarBool:
        """Loop condition: Continue until stack is empty."""
        _, _, _, pointer = state
        return pointer >= 0

    # execute the while loop
    _, final_sample, _, _ = lax.while_loop(condition, traverse_tree, initial_state)
    return final_sample


@jit
def _log_prob_single(
    initial_state: tuple,
    value: Int[jnp.ndarray, " D"],
    probs: Float[jnp.ndarray, " D"],
    is_operator: Bool[jnp.ndarray, " D"],
    max_depth: ScalarInt,
    path_to_hole: Int[jnp.ndarray, " D"],
    hole_idx: ScalarInt,
) -> ScalarFloat:
    """
    Compute the log probability of a given kernel structure.

    NOTE: This function is should be called via the log_prob_single method.

    Parameters
    ----------
    initial_state : tuple
        Initial state for the while loop.
    value :  Int[jnp.ndarray, " D"]
        An array that describes the kernel structure.
    probs :  Float[jnp.ndarray, " D"]
        The probabilities of sampling each kernel in the library.
    is_operator : Bool[jnp.ndarray, " D"]
        An array that indicates whether each kernel in the library is an operator.
    max_depth : ScalarInt
        The maximum depth of the nested kernel structure tree.
    path_to_hole :  Int[jnp.ndarray, " D"]
        The path to the hole in the tree.
    hole_idx : ScalarInt
        The index of the hole in the tree.

    Returns
    -------
    ScalarFloat
        The log probability of the given kernel structure.
    """

    def traverse_tree(
        state: tuple[ScalarFloat, Int[jnp.ndarray, " D"], ScalarInt]
    ) -> tuple[ScalarFloat, Int[jnp.ndarray, " D"], ScalarInt]:
        log_p, stack, pointer = state

        # pop the top of the stack
        pos = stack[pointer]
        pointer = pointer - 1

        # compute the depth of the current node
        depth = get_depth(pos)

        # get correspoding value from the kernel structure
        choice = value[pos]

        # determine if operators are allowed at this depth
        is_at_max_depth = depth >= max_depth
        # mask the probabilities: if at max_depth, disable operator choices
        masked_probs = jnp.where(
            is_at_max_depth,
            jnp.where(is_operator, 0.0, probs),  # if at max depth, mask operators
            probs,  # else use full probs
        )

        # determine if the current node is on path to the hole
        is_on_path = jnp.isin(pos, path_to_hole)
        masked_probs = jnp.where(
            is_on_path,
            jnp.where(
                ~is_operator, 0.0, masked_probs
            ),  # if on path, mask non-operators
            masked_probs,
        )
        masked_probs /= jnp.sum(masked_probs)

        # normalize masked_probs
        masked_probs /= jnp.sum(masked_probs)

        # update the log probability
        is_hole = pos == hole_idx
        log_p_updated = jnp.where(
            is_hole,
            log_p,
            log_p + jnp.log(masked_probs[choice]),
        )

        # determine if the choice is an operator node
        is_op = jnp.where(is_hole, False, is_operator[choice])

        def push(stack_and_pointer: tuple) -> tuple:
            """If the node is an operator, push the new
            child nodes onto the stack.
            """
            stack, pointer = stack_and_pointer

            left_child = get_child_idx(pos, "l")
            right_child = get_child_idx(pos, "r")

            stack_updated = (
                stack.at[pointer + 1].set(right_child).at[pointer + 2].set(left_child)
            )
            pointer_updated = pointer + 2
            return stack_updated, pointer_updated

        def dont_push(stack_and_pointer: tuple) -> tuple:
            """If the node is a leaf node, do not modify the stack."""
            return stack_and_pointer

        # update the stack based on whether the current node is an operator
        new_stack, new_pointer = lax.cond(
            is_op,
            push,
            dont_push,
            operand=(stack, pointer),
        )

        return (log_p_updated, new_stack, new_pointer)

    def condition(
        state: tuple[ScalarFloat, Int[jnp.ndarray, " D"], ScalarInt]
    ) -> ScalarBool:
        """Loop condition: Continue until stack is empty."""
        _, _, pointer = state
        return pointer >= 0

    # iterate over the sample using lax.while_loop
    final_log_p, _, _ = lax.while_loop(
        condition,
        traverse_tree,
        initial_state,
    )
    return jnp.array(final_log_p)
