from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from beartype.typing import Callable, List, Optional, Union
from gpjax.base import param_field, static_field
from gpjax.kernels.base import AbstractKernel
from gpjax.typing import Array, ScalarFloat
from jaxtyping import Float

# flake8: noqa: F722 # ignore typing error for jax, not supported by flake8


@dataclass
class CombinationKernel(AbstractKernel):
    r"""A base class for products or sums of MeanFunctions."""

    kernels: Optional[List[AbstractKernel]] = None
    operator: Callable = static_field(None)

    def __post_init__(self) -> None:
        # Add kernels to a list, flattening out instances of this class therein, as in GPFlow kernels.
        kernels_list: List[AbstractKernel] = []

        assert isinstance(self.kernels, list)
        for kernel in self.kernels:
            if not isinstance(kernel, AbstractKernel):
                raise TypeError("can only combine Kernel instances")  # pragma: no cover

            if isinstance(kernel, self.__class__):
                assert isinstance(kernel.kernels, list)
                kernels_list.extend(kernel.kernels)
            else:
                kernels_list.append(kernel)

        # self.kernels = kernels_list

    def __call__(
        self,
        x: Float[Array, " D"],  # type: ignore
        y: Float[Array, " D"],  # type: ignore
    ) -> ScalarFloat:
        r"""Evaluate the kernel on a pair of inputs.

        Args:
            x (Float[Array, " D"]): The left hand input of the kernel function.
            y (Float[Array, " D"]): The right hand input of the kernel function.

        Returns
        -------
            ScalarFloat: The evaluated kernel function at the supplied inputs.
        """
        assert isinstance(self.kernels, list)
        return self.operator(jnp.stack([k(x, y) for k in self.kernels]))


SumKernel = partial(CombinationKernel, operator=jnp.sum)  # type: ignore
ProductKernel = partial(CombinationKernel, operator=jnp.prod)  # type: ignore


@dataclass
class OrnsteinUhlenbeck(AbstractKernel):
    r"""The Ornstein-Uhlenbeck kernel."""

    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )
    variance: ScalarFloat = param_field(jnp.array(1.0), bijector=tfb.Softplus())
    name: str = "OU"

    def __call__(
        self,
        x: Float[Array, " D"],  # type: ignore
        y: Float[Array, " D"],  # type: ignore
    ) -> ScalarFloat:
        r"""Compute the OU kernel between a pair of arrays.

        Evaluate the kernel on a pair of inputs $`(x, y)`$ with lengthscale parameter
        $`\ell`$ and variance $`\sigma^2`$:
        ```math
        k(x,y)=\sigma^2\exp\Bigg(- \frac{\lVert x - y \rVert_2}{2 \ell^2} \Bigg)
        ```

        Args:
            x (Float[Array, " D"]): The left hand argument of the kernel
            function's call.
            y (Float[Array, " D"]): The right hand argument of the kernel
            function's call.

        Returns:
            ScalarFloat: The value of $`k(x, y)`$.
        """
        x = self.slice_input(x) / self.lengthscale
        y = self.slice_input(y) / self.lengthscale
        K = self.variance * jnp.exp(-0.5 * jnp.sum(jnp.abs(x - y)))
        return K.squeeze()
