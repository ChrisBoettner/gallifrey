from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Float, ArrayLike

from gallifrey.utils.typing import ScalarFloat, ScalarInt


class Transform(ABC):
    """Abstract base class for data transformations."""

    def __call__(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """Applies the transformation to input."""
        return self.apply(x)

    @abstractmethod
    def apply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """Applies the transformation to input."""
        pass

    @abstractmethod
    def unapply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """Unapplies the transformation to input."""
        pass

    @abstractmethod
    def apply_var(
        self,
        var_val: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """Applies the transformation to a variance value."""
        pass

    @abstractmethod
    def unapply_var(
        self,
        var_val: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """Unapplies the transformation to a variance value."""
        pass

    @classmethod
    @abstractmethod
    def from_data_range(
        cls,
        data: Float[ArrayLike, "..."],
        lo: ScalarFloat | ScalarInt,
        hi: ScalarFloat | ScalarInt,
    ) -> Transform:
        """Creates a Transform instance such that data is scaled to [lo, hi]."""
        pass

    @classmethod
    @abstractmethod
    def from_data_width(
        cls,
        data: Float[ArrayLike, "..."],
        width: ScalarFloat | ScalarInt,
    ) -> Transform:
        """Creates a Transform instance such that the width of the data
        is scaled to the given width."""
        pass


class LinearTransform(Transform):
    """
    Class for linear transformations.

    The transformation is defined as y = slope * x + intercept.

    Attributes
    ----------
    slope : float
        The slope of the linear transformation.
    intercept : float
        The intercept of the linear transformation.
    """

    def __init__(self, slope: ScalarFloat, intercept: ScalarFloat):
        """
        Initializes the LinearTransform object.

        Parameters
        ----------
        slope : ScalarFloat
            The slope of the linear transformation.
        intercept : ScalarFloat
            The intercept of the linear transformation.
        """
        self.slope = jnp.asarray(slope)
        self.intercept = jnp.asarray(intercept)

    def apply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Applies the linear transformation to input x.

        Parameters
        ----------
        x :  Float[ArrayLike, "..."]
            The input data.

        Returns
        -------
         Float[jnp.ndarray, "..."]
            The transformed data.

        """
        return self.slope * x + self.intercept

    def unapply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Unapplies the linear transformation to input x.

        Parameters
        ----------
        x :  Float[ArrayLike, "..."]
            The (reverse) transformed data.

        Returns
        -------
         Float[jnp.ndarray, "..."]
            The un-transformed data.
        """
        return jnp.asarray((x - self.intercept) / self.slope)

    def apply_mean(
        self,
        mean_val: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Applies the linear transformation to a mean value.

        Parameters
        ----------
        mean_val : Float[ArrayLike, "..."]
            The mean value to be transformed.

        Returns
        -------
        Float[jnp.ndarray, "..."]
            The transformed mean value.
        """
        return self.apply(jnp.asarray(mean_val))

    def unapply_mean(
        self,
        mean_val: Float[ArrayLike, "..."],
    ) -> Float[ArrayLike, "..."]:
        """
        Unapplies the linear transformation to a mean value.

        Parameters
        ----------
        mean_val : Float[ArrayLike, "..."]
            The mean value to be un-transformed.

        Returns
        -------
        Float[ArrayLike, "..."]
            The un-transformed mean value.
        """
        return self.unapply(jnp.asarray(mean_val))

    def apply_var(
        self,
        var_val: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Applies the linear transformation to a variance value.

        Parameters
        ----------
        var_val : Float[ArrayLike, "..."]
            The variance value to be transformed.

        Returns
        -------
        Float[jnp.ndarray, "..."]
            The transformed variance value.
        """
        return jnp.asarray(self.slope**2 * var_val)

    def unapply_var(
        self,
        var_val: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Unapplies the linear transformation to a variance value.

        Parameters
        ----------
        var_val : Float[ArrayLike, "..."]
            The variance value to be un-transformed.

        Returns
        -------
        Float[jnp.ndarray, "..."]
            The un-transformed variance value.
        """
        return jnp.asarray((1 / (self.slope**2)) * var_val)

    def apply_mean_var(
        self,
        mean_val: Float[ArrayLike, "..."],
        var_val: Float[ArrayLike, "..."],
    ) -> tuple[Float[jnp.ndarray, "..."], Float[jnp.ndarray, "..."]]:
        """
        Applies the linear transformation to mean and variance values.

        Parameters
        ----------
        mean_val : Float[ArrayLike, "..."]
            The mean value to be transformed.
        var_val : Float[ArrayLike, "..."]
            The variance value to be transformed.

        Returns
        -------
        tuple[Float[jnp.ndarray, "..."], Float[jnp.ndarray, "..."]]
            A tuple containing the transformed mean and variance values.

        """
        m = self.apply_mean(mean_val)
        v = self.apply_var(var_val)
        return (m, v)

    def unapply_mean_var(
        self,
        mean_val: Float[ArrayLike, "..."],
        var_val: Float[ArrayLike, "..."],
    ) -> tuple[Float[ArrayLike, "..."], Float[ArrayLike, "..."]]:
        """
        Unapplies the linear transformation to mean and variance values.

        Parameters
        ----------
        mean_val : Float[ArrayLike, "..."]
            The mean value to be un-transformed.
        var_val : Float[ArrayLike, "..."]
            The variance value to be un-transformed.

        Returns
        -------
        tuple[Float[ArrayLike, "..."], Float[ArrayLike, "..."]]
            A tuple containing the un-transformed mean and variance values.

        """
        m = self.unapply_mean(mean_val)
        v = self.unapply_var(var_val)
        return (m, v)

    @classmethod
    def from_data_range(
        cls,
        data: Float[ArrayLike, "..."],
        lo: ScalarFloat | ScalarInt,
        hi: ScalarFloat | ScalarInt,
    ) -> LinearTransform:
        """
        Creates a LinearTransform instance such that data
        is scaled to [lo, hi].

        NaN values are ignored in the calculation.

        Parameters
        ----------
        data :  Float[ArrayLike, "..."]
            The input data.
        lo : ScalarFloat | ScalarInt
            The lower bound of the desired range.
        hi : ScalarFloat | ScalarInt
            The upper bound of the desired

        Returns
        -------
        LinearTransform
            A LinearTransform instance with slope and intercept
            such that data is scaled to [lo, hi].

        Raises
        ------
        ValueError
            If the input data contains less than 2 non-NaN values.

        """
        tnan = jnp.asarray(data)[~jnp.isnan(data)]
        if len(tnan) < 2:
            raise ValueError("Cannot scale with <2 values.")
        tmin = jnp.min(tnan)
        tmax = jnp.max(tnan)
        a = hi - lo
        b = tmax - tmin
        slope = a / b
        intercept = -slope * tmin + lo
        return cls(slope, intercept)

    @classmethod
    def from_data_width(
        cls,
        data: Float[ArrayLike, "..."],
        width: ScalarFloat | ScalarInt,
    ) -> LinearTransform:
        """
        Creates a LinearTransform instance such that the width of the data
        is scaled to the given width, i.e., the data is scaled to
        [-width/2, width/2].

        NaN values are ignored in the calculation.

        Parameters
        ----------
        data :  Float[ArrayLike, "..."]
            The input data.
        width : ScalarFloat | ScalarInt
            The desired width of the data.

        Returns
        -------
        LinearTransform
            A LinearTransform instance with slope and intercept
            such that the data is scaled to [-width/2, width/2].

        Raises
        ------
        ValueError
            If the input data contains less than 2 non-NaN values.
        """
        tnan = jnp.asarray(data)[~jnp.isnan(data)]
        if len(tnan) < 2:
            raise ValueError("Cannot scale with <2 values.")

        a = tnan.max() - tnan.min()
        slope = width / a
        intercept = -(jnp.asarray(width) * tnan.mean()) / a
        return cls(slope, intercept)


class LogTransform(Transform):
    """
    Class for log transformations.

    The transformation is defined as y = log(x).

    """

    def apply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Applies the log transformation to input x.

        Parameters
        ----------
        x : Float[ArrayLike, "..."]
            The input data.

        Returns
        -------
        Float[jnp.ndarray, "..."]
            The transformed data.

        """

        return jnp.log(x)

    def unapply(
        self,
        x: Float[ArrayLike, "..."],
    ) -> Float[jnp.ndarray, "..."]:
        """
        Unapplies the log transformation to input x.

        Parameters
        ----------
        x : Float[ArrayLike, "..."]
            The (reverse) transformed data.

        Returns
        -------
         Float[jnp.ndarray, "..."]
            The un-transformed data.
        """
        return jnp.exp(x)

    def unapply_mean_var(
        self,
        mean_val: Float[ArrayLike, "..."],
        var_val: Float[ArrayLike, "..."],
    ) -> tuple[Float[jnp.ndarray, "..."], Float[jnp.ndarray, "..."]]:
        """
        Unapplies the log transformation to mean and variance values.

        Parameters
        ----------
        mean_val : Float[ArrayLike, "..."]
            The mean value to be un-transformed.
        var_val : Float[ArrayLike, "..."]
            The variance value to be un-transformed.

        Returns
        -------
        tuple[Float[jnp.ndarray, "..."], Float[jnp.ndarray, "..."]]
            A tuple containing the un-transformed mean and variance values.
        """
        m = jnp.exp(mean_val + var_val / 2)
        v = (jnp.exp(var_val) - 1) * jnp.exp(2 * mean_val + var_val)
        return (m, v)
