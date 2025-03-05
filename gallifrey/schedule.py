from abc import ABC, abstractmethod

import jax.numpy as jnp


class Schedule(ABC):
    """
    Abstract base class for SMC data annealing schedules.

    """

    @staticmethod
    @abstractmethod
    def generate(
        num_datapoints: int,
        num_steps: int,
        start: int = 1,
    ) -> tuple[int, ...]:
        """
        Generate an annealing schedule.

        Parameters
        ----------
        num_datapoints : int
            The total number of datapoints.
        num_steps : int
            The number of steps in the schedule.
        start : int, optional
            The starting point of the schedule,
            by default 1 (one observation).

        Returns
        -------
        tuple[int, ...]
            A tuple of integers representing the cumulative number of
            observations at each step of the schedule. Length of the
            tuple is `num_steps`.

        """
        pass


class LinearSchedule(Schedule):
    """
    Linear scheduler, adds roughly
    `n * percent` new observations at each step.

    """

    @staticmethod
    def generate(
        num_datapoints: int,
        num_steps: int,
        start: int = 1,
    ) -> tuple[int, ...]:
        """
        Generate a linear annealing schedule.

        Parameters
        ----------
        num_datapoints : int
            The total number of datapoints.
        num_steps : int
            The number of steps in the schedule.
        start : int, optional
            The starting point of the schedule,
            by default 1 (one observation).

        Returns
        -------
        tuple[int, ...]
            A tuple of integers representing the cumulative number of
            observations at each step of the schedule. Length of the
            tuple is `num_steps`.

        """
        # if only one step, run total number of datapoints
        if num_steps == 1:
            start = num_datapoints

        return tuple(
            jnp.round(
                jnp.linspace(
                    start=start,
                    stop=num_datapoints,
                    num=num_steps,
                    endpoint=True,
                    dtype=float,
                ),
            )
            .astype(int)
            .tolist()
        )


class LogSchedule(Schedule):
    """
    A logarithmic scheduler, adds observations in a logarithmic fashion.

    """

    @staticmethod
    def generate(
        num_datapoints: int,
        num_steps: int,
        start: int = 1,
    ) -> tuple[int, ...]:
        """
        Generate a logarithmic annealing schedule.

        Parameters
        ----------
        num_datapoints : int
            The total number of datapoints.
        num_steps : int
            The number of steps in the schedule.
        start : int, optional
            The starting point of the schedule,
            by default 1 (one observation).

        Returns
        -------
        tuple[int, ...]
            A tuple of integers representing the cumulative number of
            observations at each step of the schedule. Length of the
            tuple is `num_steps`.
        """
        # if only one step, run total number of datapoints
        if num_steps == 1:
            start = num_datapoints

        return tuple(
            jnp.round(
                jnp.geomspace(
                    start=start,
                    stop=num_datapoints,
                    num=num_steps,
                    endpoint=True,
                )
            )
            .astype(int)
            .tolist()
        )
