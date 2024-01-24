import dataclasses

from beartype.typing import Callable
from gpjax.base import static_field
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.typing import Array
from jaxtyping import Float, Num

# flake8: noqa: F722 # ignore typing error for jax, not supported by flake8


@dataclasses.dataclass
class Transit(AbstractMeanFunction):
    """GPJax implementation of a transit function as
    GP mean function. Takes a mean

    Parameters
    ----------
    transit_model : Callable
        The transit model, must be a function of the form
        f(x, parameter) -> y, that takes the points at which
        to evaluate the function and parameter as input, and
        returns the flux at the input points.
    transit_parameter: Array
        The transit parameter called by the transit model function.
    """

    transit_model: Callable = static_field()
    transit_parameter: Float[Array, " O"] = static_field()

    def __call__(self, x: Num[Array, "N"]) -> Float[Array, "N"]:
        """Evaluate the transit model at the given points.

        Parameters
        ----------
        x : Array
            The evaluation points.


        Returns
        -------
        Array
            The fluxes at the evaluation points
        """
        return self.transit_model(x, self.transit_parameter)
