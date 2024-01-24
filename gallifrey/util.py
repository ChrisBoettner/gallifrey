import jax.numpy as jnp
import jax.random as jr
from beartype.typing import Any
from jax import jit
from jaxoplanet import orbits
from jaxoplanet.light_curves import LimbDarkLightCurve
from jaxtyping import Array

rng_key = jr.PRNGKey(36)
idx_key, noise_key = jr.split(rng_key, 2)


@jit
def example_transit_model(t: Array, params: list) -> Array:
    """Example transit model using jaxoplanet.

    Parameters
    ----------
    t : Array
        The time's at which to evaluate the light curve.
    params : list
        The light curve parameter (radius, u1, u2).

    Returns
    -------
    Array
        The example transit light curve.
    """
    orbit = orbits.keplerian.Body(
        period=15,
        radius=params[0],
        inclination=jnp.deg2rad(89),
        time_transit=0,
    )

    lc = LimbDarkLightCurve([params[1], params[2]]).light_curve(orbit, t=t)
    return lc


def example_lightcurve(
    noise_std: float = 0.001,
    correlated: bool = False,
    num_train: int = 150,
) -> dict:
    """Calculate a toy example light curve.

    Parameters
    ----------
    noise_std : float, optional
        White noise std, by
        default 0.001.
    phi : float, optional
        If True, create AR(1) noise with phi=0.8, rather
        than white noise. By default False.
    num_train : int, optional
        Number of tranings points (<1000), by
        default 150.

    Returns
    -------
    dict
        Dictonary containing the toy example.
    """
    example: dict[str, Any] = {}

    example["white_noise_std"] = noise_std
    if correlated:
        example["phi"] = 0.8
        example["name"] = "gpmodel_ar_noise"
    else:
        example["phi"] = 0
        example["name"] = "gpmodel"

    example["t"] = jnp.linspace(-0.8, 0.8, 1000)

    example["transit"] = example_transit_model(example["t"], [0.1, 0.1, 0.3])

    example["background"] = 0.002 * (
        5 * example["t"] ** 2
        + jnp.sin(20 * example["t"])
        + 0.3 * jnp.cos(50 * example["t"])
    )

    white_noise = noise_std * jr.normal(noise_key, (len(example["t"]),))
    noise = white_noise
    noise = jnp.zeros(len(example["t"]))
    noise = noise.at[0].set(white_noise[0])
    for i in range(1, len(example["t"])):  # generate ar(1) noise
        noise = noise.at[i].set(example["phi"] * noise[i - 1] + white_noise[i])
    example["noise"] = noise
    example["noise_std"] = noise_std / jnp.sqrt(1 - example["phi"] ** 2)

    train_ind = jnp.sort(
        jr.choice(idx_key, len(example["t"]), (num_train,), replace=False)
    )

    example["t_sample"] = example["t"][train_ind]
    example["lc_sample"] = (
        example["transit"] + example["background"] + example["noise"]
    )[train_ind]

    example["transit_mask"] = jnp.isclose(example["transit"], 0)
    example["transit_mask_sample"] = example["transit_mask"][train_ind]
    return example
