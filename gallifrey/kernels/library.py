import warnings

import beartype.typing as tp
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from flax.core import FrozenDict

from gallifrey.kernels.atoms import (
    AbstractAtom,
    AbstractOperator,
    LinearAtom,
    PeriodicAtom,
    ProductOperator,
    RBFAtom,
    SumOperator,
)


class KernelLibrary:
    """
    A class to store the kernel library, which contains information
    about what atoms and operators are available for constructing
    the tree kernel.

    Attributes
    ----------
    atoms : tp.Optional[list[AbstractAtom]]
        A list of atoms that are used to construct the tree kernel.
        If None, the default atoms are used. Atoms should inherit
        from the AbstractAtom class in gallifrey.kernels.atoms.
    operators : tp.Optional[list[AbstractOperator]]
        A list of operators that are used to combine the atoms in the
        tree kernel. If None, the default operators are used. Operators
        should inherit from the AbstractOperator class in
        gallifrey.kernels.atoms.
    num_atoms : int
        The number of atoms in the library.
    num_operators : int
        The number of operators in the library.
    library : list[tp.Union[AbstractAtom, AbstractOperator]]
        A list combining the atoms and operators in the library.
    is_operator : jnp.ndarray
        A boolean array to indicate if an entry in the library is an operator
        or not.
    max_atom_parameters : int
        The maximum number of parameters any atom or operator in the library
        takes as input.
    prior_transforms : tp.Optional[dict[str, tfb.Bijector]]
        A dictionary that maps the parameter tags to the corresponding
        transform functions. Transformation functions must transform
        parameter from a standard normal distribution to the desired
        (prior) distribution, and should be implemented via
        tensorflow_probability bijectors. The default transformations
        are used if None. The default transformations are:
        - "positive": Log-normal transformation(mu=0.0, sigma=1.0)
        - "real": Log-normal transformation(mu=0.0, sigma=1.0)
        - "sigmoid": Logit-normal transform(scale=1.0, mu=0.0, sigma=1.0)
        NOTE: To make functions that use this dict jittable, the prior_transforms
        is converted to a FrozenDict, to make it hashable.
        NOTE: The "none" key is reserved for internal use and should not be
        used in the prior_transforms dictionary.
    support_tag_mapping : dict[str, int]
        A dictionary that maps the support tags to integer values.
    support_mapping_array : jnp.ndarray
        An array that contains the support tags as integers for each parameter
        in the library.
    support_transforms : FrozenDict[str, Transformation]
        A (frozen) dict that maps the parameter tags to the corresponding transform
        functions. Transformation functions must transform parameter from a constrained
        space to an unconstrained space. This is used for sampling and optimization.
        NOTE: This should ideally never be touched, unless you know what you are doing.

    """

    def __init__(
        self,
        atoms: tp.Optional[list[AbstractAtom]] = None,
        operators: tp.Optional[list[AbstractOperator]] = None,
        prior_transforms: tp.Optional[dict[str, tfb.Bijector]] = None,
    ):
        """
        Initialize the KernelLibrary. If no atoms or operators are provided,
        a default library is used.

        Parameters
        ----------
        atoms : tp.Optional[list[AbstractAtom]], optional
            A list of atoms that are used to construct the tree kernel.
            If None, the default atoms are used. Atoms should inherit
            from the AbstractAtom class in gallifrey.kernels.atoms, by default None.
        operators : tp.Optional[list[AbstractOperator]], optional
            A list of operators that are used to combine the atoms in the
            tree kernel. If None, the default operators are used. Operators
            should inherit from the AbstractOperator class in
            gallifrey.kernels.atoms, by default None.
        prior_transforms : tp.Optional[dict[str, Transformation]], optional
            A dictionary that maps the parameter tags to the corresponding transform
            functions. Transformation functions must transform parameter from a standard
            normal distribution to the desired (prior) distribution, and should be
            implemented via tensorflow_probability bijectors.
            The default transformations are used if None, by default None. The default
            transformations are:
            - "positive": Log-normal transformation(mu=0.0, sigma=1.0)
            - "real": Log-normal transformation(mu=0.0, sigma=1.0)
            - "sigmoid": Logit-normal transform(scale=1.0, mu=0.0, sigma=1.0)
            NOTE: To make 'transformation' function (see gallifrey.kernels.prior)
            jitable, the prior_transforms must be hashable. This is why we use
            FrozenDict instead of dict.

        """
        self.atoms: list[AbstractAtom] = (
            atoms
            if atoms is not None
            else [
                LinearAtom(),
                PeriodicAtom(),
                RBFAtom(),
            ]
        )

        self.operators: list[AbstractOperator] = (
            operators
            if operators is not None
            else [
                SumOperator(),
                ProductOperator(),
            ]
        )

        self.num_atoms = len(self.atoms)
        self.num_operators = len(self.operators)

        # construct the library by combining the atoms and operators
        self.library = self.atoms + self.operators

        # create boolean array to indicate if entry is operator or not
        self.is_operator = jnp.array(
            [False] * len(self.atoms) + [True] * len(self.operators)
        )

        # get the maximum number of parameters any atom or operator takes
        self.max_atom_parameters = max([item.num_parameter for item in self.atoms])

        # transformation functions from normal distribution to prior distribution
        # NOTE: names are inherited from GPJax, but we don't use the same
        # transformations (which is why sigmoid is the name for the logit-normal)
        if prior_transforms is None:
            self.prior_transforms: FrozenDict[str, tfb.Bijector] = FrozenDict(
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
                                high=jnp.array(0.95),  # to avoid numerical issues
                            ),
                            tfb.Shift(jnp.array(0.0)),
                            tfb.Scale(jnp.array(0.0)),
                        ]
                    ),
                    # identity transformation for parameter that do not fall under
                    # the above categories
                    "none": tfb.Identity(),
                }
            )

        else:
            if "none" in prior_transforms.keys():
                raise ValueError(
                    "'prior_transforms' should not contain the key 'none', "
                    "as it is reserved for internal use."
                )
            prior_transforms["none"] = tfb.Identity()
            self.prior_transforms = FrozenDict(prior_transforms)

        self.support_mapping_array, self.support_tag_mapping = (
            self.get_support_mapping()
        )

        # besides the prior transforms, we also perform transforms
        # between a constrained and unconstrained space for fitting/sampling,
        # this probably should never be touched unless you implement a new
        # kernel with new constraints
        self.support_transforms: FrozenDict[str, tfb.Bijector] = FrozenDict(
            {
                "positive": tfb.Softplus(),
                "real": tfb.Identity(),
                "sigmoid": tfb.Sigmoid(low=0.0, high=0.95),
                "lower_triangular": tfb.FillTriangular(),
                "none": tfb.Identity(),
            }
        )

        self._check_tags()

    def __repr__(self) -> str:
        """
        Get the (technical) string representation of the KernelLibrary.

        Returns
        -------
        str
            The (technical) string representation of the KernelLibrary.
        """

        return (
            f"KernelLibrary(\n  Atoms={self.atoms},\n   "
            f"operators={self.operators}\n)"
        )

    def __str__(self) -> str:
        """
        Get the (simplified) string representation of the KernelLibrary.

        Returns
        -------
        str
            The (simplified) string representation of the KernelLibrary.
        """
        try:
            atom_names: list[tp.Any] = [atom.name for atom in self.atoms]
        except Exception:
            atom_names = self.atoms
        try:
            operator_names: list[tp.Any] = [
                operator.name for operator in self.operators
            ]
        except Exception:
            operator_names = self.operators

        return (
            f"KernelLibrary(\n  Atoms={atom_names},\n   "
            f"operators={operator_names}\n)"
        )

    def __len__(self) -> int:
        """
        Get the number of items in the library.

        Returns
        -------
        int
            The number of items in the library.
        """
        return len(self.library)

    def get_support_mapping(self) -> tuple[jnp.ndarray, dict[str, int]]:
        """
        Create a mapping between support tags and the integer values, then
        create an array that contains the support tags for each parameter
        in the library.
        (Used for jittable sampling and parameter transformations)

        Returns
        -------
        support_mapping_array : jnp.ndarray
            An array that contains the support tags for each parameter
            in the library, with shape (len(self), self.max_atom_parameters),
            enocded as integers.
        support_tag_mapping : dict[str, int]
            A dictionary that maps the support tags to the corresponding
            integer values.
        """

        # create dict that maps string tags to integer values
        support_tag_mapping = {
            support_tag: i for i, support_tag in enumerate(self.prior_transforms.keys())
        }
        support_tag_mapping["none"] = -1

        # create array that contains the support tag integers for each parameter
        # in the library
        support_mapping_array = jnp.full(
            (len(self), self.max_atom_parameters), support_tag_mapping["none"]
        )

        for i in range(len(self)):
            for j in range(self.library[i].num_parameter):
                support_mapping_array = support_mapping_array.at[i, j].set(
                    support_tag_mapping[self.library[i].parameter_support[j]]
                )
        return support_mapping_array, support_tag_mapping

    def _check_tags(self) -> None:
        """
        Check if the parameters of the atoms have constraints that are not yet
        implemented in the library. If so, raise a warning.

        """
        parameter_support_tags = {
            atom.name: atom.parameter_support for atom in self.atoms
        }

        for atom_name, tags in parameter_support_tags.items():
            if not all(tag in self.prior_transforms.keys() for tag in tags):
                warnings.warn(
                    f"Atom {atom_name!r} has parameter tags that are not "
                    "in the prior_transforms dictionary. Tag will be ignored. "
                    "This will lead to problems when performing parameter "
                    "transformations. Add the missing tags to the prior_transforms"
                    "by manually passing the prior_transforms dictionary to the "
                    "KernelLibrary constructor."
                )

        if not all(tag in self.support_transforms.keys() for tag in tags):
            warnings.warn(
                f"Kernel {atom_name!r} has parameter tags that are not "
                "in the support_transforms dictionary. This will lead to problems "
                "in the sampling/optmization if used for a KernelTree object. "
                "Add the missing tags to the by manually overriding the "
                "support_transforms dictionary to the KernelLibrary constructor.\n"
                "THIS IS VERY EXPERIMENTAL AND NOT RECOMMENDED."
            )
