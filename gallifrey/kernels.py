from dataclasses import dataclass, fields
from functools import partial

import jax.numpy as jnp
from beartype.typing import Callable, List, Optional
from gpjax.base import static_field
from gpjax.gps import AbstractLikelihood
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


def flatten_kernels(kernels: list[AbstractKernel]) -> list[AbstractKernel]:
    """Flatten a (nested) list of kernels and Combination kernels into
    the base kernels.

    Parameters
    ----------
    kernels : list[AbstractKernel]
        The (nested) list of kernels.

    Returns
    -------
    list[AbstractKernel]
        The flattened list of kernels.
    """
    flattened = []
    for kernel in kernels:
        if isinstance(kernel, CombinationKernel):
            # Recursively flatten the list
            flattened.extend(flatten_kernels(kernel.kernels))  # type: ignore
        else:
            # It's an base AbstractKernel but not a CombinationKernel
            flattened.append(kernel)
    return flattened


def get_kernel_info(kernel: AbstractKernel | AbstractLikelihood) -> list:
    """Get kernel parameter names, values and
    trainable status.

    Parameters
    ----------
    kernel : AbstractKernel
        The kernel (or likelihood for std parameter).

    Returns
    -------
    list
        A list containing tuples with
        (parameter name, parameter value, trainable).
    """
    kernel_dict = vars(kernel)
    meta_data = kernel_dict.get("_pytree__meta")

    kernel_info = []
    if meta_data is not None:
        # Process using _pytree__meta
        for field_name, field_value in kernel_dict.items():
            if field_name == "_pytree__meta":
                continue

            field_meta = meta_data.get(field_name, {})
            trainable = field_meta.get("trainable")
            if trainable is not None:
                kernel_info.append((field_name, field_value, trainable))
    else:
        # Process using dataclasses fields
        for f in fields(kernel):
            if f.metadata.get("trainable") is not None:
                kernel_info.append(
                    (f.name, getattr(kernel, f.name), f.metadata["trainable"])
                )
    return kernel_info
