import warnings

import jax.numpy as jnp
from jaxtyping import install_import_hook

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

from gallifrey.gps.kernels import (
    CombinationKernel,
    flatten_kernels,
    get_kernel_info,
)
from gallifrey.gps.trainables import get_trainables


def _describe_kernel(
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
            sub_names = [_describe_kernel(sub_k) for sub_k in k.kernels]
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


def kernel_summary(
    model: gpx.kernels.AbstractKernel
    | gpx.gps.AbstractPosterior
    | gpx.gps.AbstractPrior,
    to_latex: bool = False,
    silence: bool = False,
    short: bool = False,
) -> str:
    """
    Constructs and returns a string describing the model as
    determined by kernel search. If to_latex=True, returns string in
    LateX table format.

    Works with nested CombinationKernels in the kernel tree, but only
    those created explicitly by the kernel search and its particular
    structure.

    Parameters
    ----------
    kernel :  gpx.kernels.AbstractKernel
            | gpx.gps.AbstractPosterior
            | gpx.gps.AbstractPrior
        The kernel to be described. Can also pass posterior or
        prior object, in which case the associated kernel is
        described.
    to_latex : bool, optional
        If True, returns a LaTeX table format of the kernel summary,
        by default False.
    silence: bool, optional
        If True, don't print output, by default False.
    short: bool, optional
        If True, only returns kernel architecture without description
        of the parameter, by default False.

    Returns
    -------
    str
        A string containing the summary of the kernel, either as
        plain text or a LaTeX table.
    """
    likelihood_info = None
    if isinstance(model, gpx.gps.AbstractPosterior):
        if isinstance(model.likelihood, gpx.gps.Gaussian):
            likelihood_info = get_kernel_info(model.likelihood)[0]
        else:
            warnings.warn(
                "Only parameters for Gaussian likelihood are currently printed.",
                stacklevel=2,
            )
        kernel = model.prior.kernel
    elif isinstance(model, gpx.gps.AbstractPrior):
        kernel = model.kernel
    elif isinstance(model, gpx.kernels.AbstractKernel):
        kernel = model
    else:
        raise ValueError("'kernel' must be kernel, prior or posterior instance.")
    assert isinstance(kernel, gpx.kernels.AbstractKernel)

    kernel_description = _describe_kernel(kernel)
    kernel_description = (
        kernel_description.replace("•", r"$\cdot$") if to_latex else kernel_description
    )

    if short:
        output = kernel_description

    else:
        if hasattr(kernel, "kernels"):
            kernels = flatten_kernels(kernel.kernels)  # type: ignore
        else:
            kernels = [kernel]

        output = ""
        if to_latex:
            output += "\\begin{table}[ht]\n"
            output += "\\centering\n"
            caption = kernel_description
            if likelihood_info:
                caption += (
                    f" with observed stddev = {likelihood_info[1]:.5e} "
                    f"(Trainable : {likelihood_info[2]})"
                )
            output += "\\caption{Kernel Summary: " + caption + "}\n"
            output += "\\begin{tabular}{llll}\n"
            output += "Kernel & Property " "& Value & Trainable \\\\\n"
            output += "\\hline\\hline\n"
        else:
            # Header
            output += "Kernel Summary\n\n"
            output += f"Number of Parameter: {len(get_trainables(kernel))}\n"
            output += "=" * 80 + "\n"

            # Kernel description
            kernel_description = _describe_kernel(kernel)
            output += f"Kernel Structure: {kernel_description}\n"
            if likelihood_info:
                output += (
                    f"  with {likelihood_info[0]} = {likelihood_info[1]:.5e} "
                    f"(Trainable : {likelihood_info[2]})\n\n"
                )

            # Column headers
            output += (
                f"{'Kernel':<20} {'Property':<20} {'Value':<20} {'Trainable':<10}\n"
            )
            output += "-" * 80 + "\n"

        # Individual kernel properties
        for k in kernels:
            kernel_info = get_kernel_info(k)
            if kernel_info:
                for idx, (name, value, trainable) in enumerate(kernel_info):
                    formatted_value = f"{value:.5e}"
                    if to_latex:
                        if idx == 0:
                            output += f"{k.name} & "
                        else:
                            output += " & "
                        output += f"{name} & {formatted_value} & {trainable} \\\\\n"
                    else:
                        if idx == 0:
                            output += f"{k.name:<20}"
                        else:
                            output += " " * 20
                        output += (
                            f"{name:<20} {formatted_value:<20} {str(trainable):<10}\n"
                        )
                        if idx < len(kernel_info) - 1:
                            output += "\n"
                if to_latex:
                    output += "\\hline\n"
                else:
                    output += "-" * 80 + "\n"

        if to_latex:
            output += "\\end{tabular}"
            output += "\\end{table}"

    if not silence:
        print(output)
    return output
