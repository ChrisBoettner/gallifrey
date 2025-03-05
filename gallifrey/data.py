import jax.numpy as jnp
from jaxtyping import Float


class Dataset:
    """
    A class to hold the data.

    Attributes
    ----------
    x : Float[jnp.ndarray, " D"]
        The input x data (1D).
    y : Float[jnp.ndarray, " D"]
        The input y data (1D).
    n : int
        The number of data points.

    """

    def __init__(
        self,
        x: Float[jnp.ndarray, " D"],
        y: Float[jnp.ndarray, " D"],
    ):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        x : Float[jnp.ndarray, " D"]
            The input x data.
        y : Float[jnp.ndarray, " D"]
            The input y data.
        """
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)

        self._validate_input()

    @property
    def n(self) -> int:
        """
        Get the number of data points.

        Returns
        -------
        int
            The number of data points.
        """
        return self.x.shape[0]

    def _validate_input(self) -> None:
        """
        Check that the number of x and y values are the same.

        Raises
        ------
        ValueError
            If the number of x and y values are not the same.
        """

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("The length of x and y must be the same.")

    def __repr__(self) -> str:
        """
        Return the string representation of the Dataset instance.

        Returns
        -------
        str
            The representation of the Dataset instance.
        """
        return f"Dataset(n={self.n},\n x={self.x},\n y={self.y})"

    def __str__(self) -> str:
        """
        Simplified string representation of the Dataset instance.

        Returns
        -------
        str
            The simplified string representation.
        """
        return f"Dataset(n={self.n})"
