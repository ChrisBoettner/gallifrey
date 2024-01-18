import jax.numpy as jnp
import jax.random as jr
from beartype.typing import Optional
from jaxoplanet import orbits
from jaxoplanet.light_curves import LimbDarkLightCurve
from jaxtyping import Array

rng_key = jr.PRNGKey(36)
idx_key, noise_key = jr.split(rng_key, 2)


def calculate_example_lightcurve(
    t: Optional[Array] = None,
    noise_std: float = 0.001,
    phi: float = 0,
    num_train: int = 150,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    if t is None:
        t = jnp.linspace(-0.8, 0.8, 1000)

    # The light curve calculation requires an orbit
    orbit = orbits.keplerian.Body(
        period=15,
        radius=0.1,
        inclination=jnp.deg2rad(89),
        time_transit=0,
    )

    lightcurve = LimbDarkLightCurve([0.1, 0.3]).light_curve(orbit, t=t)

    systematics = 0.002 * (5 * t**2 + jnp.sin(20 * t) + 0.3 * jnp.cos(50 * t))

    white_noise = noise_std * jr.normal(noise_key, (len(t),))
    noise = white_noise
    noise = jnp.zeros(len(t))
    noise = noise.at[0].set(white_noise[0])
    for i in range(1, len(t)):  # generate ar(1) noise
        noise = noise.at[i].set(phi * noise[i - 1] + white_noise[i])

    train_ind = jnp.sort(jr.choice(idx_key, len(t), (num_train,), replace=False))

    t_train = t[train_ind]
    lc_train = (lightcurve + systematics + noise)[train_ind]

    mask = jnp.isclose(lightcurve, 0)
    train_mask = mask[train_ind]

    return (
        t_train,
        lc_train,
        train_mask,
        t,
        lightcurve,
        systematics,
        noise,
        mask,
    )
