"""Custom sources for FDTDX antenna simulations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.sources.source import Source


@autoinit
class GapVoltageSource(Source):
    """Lumped-gap E-field source for antenna feeding."""

    polarization_axis: int = frozen_field(default=2)
    amplitude: float = frozen_field(default=1.0)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities
        if inverse:
            return E
        time = time_step * self._config.time_step_duration
        amp = self.temporal_profile.get_amplitude(
            time,
            period=self.wave_character.get_period(),
            phase_shift=self.wave_character.phase_shift,
        )
        amp = amp * self.static_amplitude_factor * self.amplitude
        shape = (3,) + self.grid_shape
        update = jnp.zeros(shape, dtype=E.dtype)
        update = update.at[self.polarization_axis].set(jnp.ones(self.grid_shape, dtype=E.dtype) * amp)
        return E.at[:, *self.grid_slice].add(update)

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities, time_step
        if inverse:
            return H
        return H
