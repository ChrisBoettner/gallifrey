from abc import ABC, abstractmethod
from typing import Literal

import jax.numpy as jnp
from jax import vmap
from jaxtyping import Float


class AbstractAtom(ABC):
    """
    Abstract base class for defining atom kernels
    (basic kernel functions).

    Atoms are the fundamental building blocks for constructing more complex kernel
    structure. This class defines the interface that all atom kernels must
    implement within the `gallifrey` library.

    Attributes
    ----------
    name : str
        A descriptive name for the atom kernel (e.g., "RBF", "Linear"). This name
        is used for identification and representation purposes.
    num_parameter : int
        The number of trainable parameters associated with this atom kernel.
        For example, an RBF kernel might have a lengthscale and a variance parameter,
        so `num_parameter` would be 2.
    parameter_support : list[Literal["positive", "real", "sigmoid", "none"]]
        A list defining the support constraints for each parameter of the atom kernel.
        Each element in the list corresponds to a parameter and specifies its
        constraint:
        - `"positive"`: Parameter must be positive (e.g., lengthscale, variance).
        - `"real"`: Parameter can be any real number (unconstrained).
        - `"sigmoid"`: Parameter is constrained to the range (0, 1) via a sigmoid
          function.
        - `"none"`: No specific constraint is applied.
        The length of this list should be equal to `num_parameter`.
    parameter_names : list[str]
        A list of names for each parameter of the atom kernel. These names should be
        descriptive and correspond to the parameters in the order defined.
        The length of this list should be equal to `num_parameter`.

    Methods
    -------
    __call__(x, y, params)
        Abstract method to compute the kernel value between two input points `x`
        and `y`. This method must be implemented by concrete atom kernel classes.
    __str__()
        Returns the name of the atom kernel as a string.
    cross_covariance(x, y, params)
        Computes the cross-covariance matrix between two sets of input vectors `x` and
        `y`. Innput must be a 1D vector. The output is a 2D matrix.
        This method uses `jax.vmap` for efficient computation over batched inputs.
    gram(x, params)
        Computes the Gram matrix (covariance matrix) for a vector of input points `x`.
        This is a special case of `cross_covariance` where both input sets are the same.

    """

    name: str
    num_parameter: int
    parameter_support: list[Literal["positive", "real", "sigmoid", "none"]]
    parameter_names: list[str]

    @abstractmethod
    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        pass

    def __str__(self) -> str:
        return self.name

    def cross_covariance(
        self,
        x: Float[jnp.ndarray, " M"] | Float[jnp.ndarray, "M 1"],
        y: Float[jnp.ndarray, " N"] | Float[jnp.ndarray, "N 1"],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "M N"]:
        return vmap(lambda x_: vmap(lambda y_: self(x_, y_, params))(y))(x)

    def gram(
        self,
        x: Float[jnp.ndarray, " D"] | Float[jnp.ndarray, "D 1"],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "D D"]:
        return self.cross_covariance(x, x, params)


class AbstractOperator(ABC):
    """
    An abstract class to define the operators (combination
    functions) in the kernel structure. The operators are
    used to combine the atoms in the kernel computations.

    """

    name: str
    num_parameter: int = 0
    parameter_support: list = []
    parameter_names: list = []

    @abstractmethod
    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        pass


class SumOperator(AbstractOperator):
    """Sum Operator to combine two kernel functions."""

    name = "+"

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        return jnp.asarray(x + y).squeeze()


class ProductOperator(AbstractOperator):
    """Product Operator to combine two kernel functions."""

    name = "*"

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        return jnp.asarray(x * y).squeeze()


class ConstantAtom(AbstractAtom):
    """Constant atom."""

    name = "Constant"
    num_parameter = 1
    parameter_support = ["real"]
    parameter_names = ["constant"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        constant = params[0]
        return jnp.asarray(constant).astype(jnp.asarray(x).dtype).squeeze()


class RBFAtom(AbstractAtom):
    """Radial Basis Function/Squared Exponential atom."""

    name = "RBF"
    num_parameter = 2
    parameter_support = ["positive", "positive"]
    parameter_names = ["lengthscale", "variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        d_squared = jnp.square(x - y) / jnp.square(lengthscale)
        k = variance * jnp.exp(-0.5 * d_squared)
        return k.squeeze()


class Matern12Atom(AbstractAtom):
    """Matern 1/2 atom."""

    name = "Matern12"
    num_parameter = 2
    parameter_support = ["positive", "positive"]
    parameter_names = ["lengthscale", "variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        tau = jnp.abs(x - y) / lengthscale
        k = variance * jnp.exp(-tau)
        return k.squeeze()


class Matern32Atom(AbstractAtom):
    """Matern 3/2 atom."""

    name = "Matern32"
    num_parameter = 2
    parameter_support = ["positive", "positive"]
    parameter_names = ["lengthscale", "variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        tau = jnp.sqrt(3.0) * jnp.abs(x - y) / lengthscale
        k = variance * (1.0 + tau) * jnp.exp(-tau)
        return k.squeeze()


class Matern52Atom(AbstractAtom):
    """Matern 5/2 atom."""

    name = "Matern52"
    num_parameter = 2
    parameter_support = ["positive", "positive"]
    parameter_names = ["lengthscale", "variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        tau = jnp.sqrt(5.0) * jnp.abs(x - y) / lengthscale
        k = variance * (1.0 + tau + jnp.square(tau) / 3.0) * jnp.exp(-tau)
        return k.squeeze()


class PeriodicAtom(AbstractAtom):
    """Periodic atom."""

    name = "Periodic"
    num_parameter = 3
    parameter_support = ["positive", "positive", "positive"]
    parameter_names = ["lengthscale", "variance", "period"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        period = params[2]
        sine_squared = (jnp.sin(jnp.pi * (x - y) / period) / lengthscale) ** 2
        k = variance * jnp.exp(-0.5 * sine_squared)
        return k.squeeze()


class PoweredExponentialAtom(AbstractAtom):
    """Powered Exponential atom."""

    name = "PoweredExponential"
    num_parameter = 3
    parameter_support = ["positive", "positive", "sigmoid"]
    parameter_names = ["lengthscale", "variance", "power"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        power = params[2]
        tau = jnp.abs(x - y) / lengthscale
        k = variance * jnp.exp(-jnp.power(tau, power))
        return k.squeeze()


class RationalQuadraticAtom(AbstractAtom):
    """Rational Quadratic atom."""

    name = "RationalQuadratic"
    num_parameter = 3
    parameter_support = ["positive", "positive", "positive"]
    parameter_names = ["lengthscale", "variance", "alpha"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        lengthscale = params[0]
        variance = params[1]
        alpha = params[2]
        tau = jnp.abs(x - y) / lengthscale
        k = variance * (1 + 0.5 * jnp.square(tau) / alpha) ** (-alpha)
        return k.squeeze()


class WhiteAtom(AbstractAtom):
    """White atom."""

    name = "White"
    num_parameter = 1
    parameter_support = ["positive"]
    parameter_names = ["variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        variance = params[0]
        k = jnp.equal(x, y) * variance
        return k.squeeze()


class LinearAtom(AbstractAtom):
    """(Non-stationary) linear atom."""

    name = "Linear"
    num_parameter = 1
    parameter_support = ["positive"]
    parameter_names = ["variance"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        variance = params[0]
        k = variance * (x * y)
        return jnp.asarray(k).squeeze()


class LinearWithShiftAtom(AbstractAtom):
    """(Non-stationary) linear atom."""

    name = "Linear"
    num_parameter = 3
    parameter_support = ["positive", "positive", "real"]
    parameter_names = ["bias", "variance", "shift"]

    def __call__(
        self,
        x: Float[jnp.ndarray, "..."],
        y: Float[jnp.ndarray, "..."],
        params: Float[jnp.ndarray, " P"],
    ) -> Float[jnp.ndarray, "..."]:
        bias = params[0]
        variance = params[1]
        shift = params[2]
        k = bias + variance * ((x - shift) * (y - shift))
        return jnp.asarray(k).squeeze()
