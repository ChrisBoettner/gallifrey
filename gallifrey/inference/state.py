from __future__ import annotations

import beartype.typing as tp
from flax import struct


@struct.dataclass
class GPState:
    """
    A dataclass to hold the state of the SMC or
    MCMC algorithm.

    Attributes
    ----------
    particle_states : nnx.State
        The particle states (batched into a single State object).
    num_particles : Int[ArrayLike, "..."]
        The number of particles.
    num_data_points : Int[ArrayLike, "..."]
        The number of data points used in the SMC round. (All data
        points are used in the MCMC algorithm.)
    mcmc_accepted : Int[ArrayLike, "..."]
        The number of accepted MCMC steps per particle.
    hmc_accepted : Int[ArrayLike, "..."]
        The number of accepted HMC steps per particle.
    log_weights : Float[ArrayLike, "..."]
        The log weights of the particles (normalised). (Only
        used in the SMC algorithm, default is None.)
    marginal_log_likelihoods : Float[ArrayLike, "..."]
        The marginal log likelihoods of the particles, at the current
        point in the anneaing schedule. (Only used in the SMC
        algorithm, default is None.)
    resampled : Bool[ArrayLike, "..."]
        Whether the particles have been resampled in the current round.
        (Only used in the SMC algorithm, default is None.)
    key : tp.Optional[PRNGKeyArray | ArrayLike]
        The random key for this round. (Only used in the SMC
        algorithm, default is None.)

    """

    # overly permissive type hints to allow for checkpointing
    # TODO: find a better way to handle this

    particle_states: tp.Any
    num_particles: tp.Any
    num_data_points: tp.Any  # num data points used in the SMC round
    mcmc_accepted: tp.Any
    hmc_accepted: tp.Any
    log_weights: tp.Any = None
    marginal_log_likelihoods: tp.Any = None
    resampled: tp.Any = None
    key: tp.Any = None

    @classmethod
    def from_dict(cls, state_dict: dict) -> GPState:
        """
        Create a GPState object from a dictionary.

        Parameters
        ----------
        state_dict : dict
            The dictionary containing the state. Must contain keys
            matching the attributes of the GPState class.
        """
        return GPState(**state_dict)
