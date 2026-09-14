"""Semaglutide 1-compartment PK/PD ODEs integrated with Diffrax.

Time unit is days. Depot amount is in mg; plasma concentration C is in mg/L.

The plasma ODE matches the project spec when ``v_d = 1`` (Depot already in
concentration units). With the default ``v_d`` (L), the standard first-order
absorption form is used:

    d Depot / dt = -k_a * Depot
    d C / dt     = -k_e * C + k_a * Depot / v_d

Elimination uses ``k_e = ln(2) / t_half`` with ``t_half ≈ 7`` days.
"""

from __future__ import annotations

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class PKPDParams(eqx.Module):
    """Fixed biophysical parameters for semaglutide PK/PD."""

    k_a: Float[Array, ""]
    k_e: Float[Array, ""]
    v_d: Float[Array, ""]
    bioavailability: Float[Array, ""]
    e_max: Float[Array, ""]
    e_c50: Float[Array, ""]
    hill: Float[Array, ""]

    @classmethod
    def semaglutide(cls) -> PKPDParams:
        """Literature-scale defaults for once-weekly SC semaglutide."""
        t_half = 7.0
        return cls(
            k_a=jnp.asarray(0.75), # absorption rate constant
            k_e=jnp.asarray(jnp.log(2.0) / t_half), # elimination rate constant
            v_d=jnp.asarray(12.5), # volume of distribution # TODO: can be made a function of body weight
            bioavailability=jnp.asarray(0.89), # fraction of dose absorbed
            e_max=jnp.asarray(1.0), # maximum effect
            e_c50=jnp.asarray(0.05), # concentration at which 50% of the maximum effect is achieved
            hill=jnp.asarray(1.0), # hill coefficient
        )


class PKPDState(eqx.Module):
    """Latent PK state: subcutaneous depot (mg) and plasma C (mg/L)."""

    depot: Float[Array, ""]
    concentration: Float[Array, ""]

    @classmethod
    def zeros(cls) -> PKPDState:
        return cls(depot=jnp.asarray(0.0), concentration=jnp.asarray(0.0))

    def as_vector(self) -> Float[Array, "2"]:
        return jnp.stack([self.depot, self.concentration])

    @classmethod
    def from_vector(cls, y: Float[Array, "2"]) -> PKPDState:
        return cls(depot=y[0], concentration=y[1])


def pk_vector_field(
    t: Float[Array, ""],
    y: Float[Array, "2"],
    args: PKPDParams,
) -> Float[Array, "2"]:
    """Right-hand side of the 1-compartment first-order absorption ODE."""
    del t
    depot, concentration = y[0], y[1]
    d_depot = -args.k_a * depot
    d_concentration = -args.k_e * concentration + args.k_a * depot / args.v_d
    return jnp.stack([d_depot, d_concentration])


def emax_pd(concentration: Float[Array, ""], params: PKPDParams) -> Float[Array, ""]:
    """Saturating GLP-1R effect in [0, e_max] from plasma concentration."""
    c_n = jnp.power(jnp.maximum(concentration, 0.0), params.hill)
    ec50_n = jnp.power(params.e_c50, params.hill)
    return params.e_max * c_n / (ec50_n + c_n)


def inject_dose(state: PKPDState, dose_mg: Float[Array, ""], params: PKPDParams) -> PKPDState:
    """Instantaneous SC bolus: add ``F * dose`` to the depot (pure)."""
    absorbed = params.bioavailability * dose_mg
    return PKPDState(
        depot=state.depot + absorbed,
        concentration=state.concentration,
    )


def integrate_pk(
    state: PKPDState,
    params: PKPDParams,
    t0: float | Float[Array, ""] = 0.0,
    t1: float | Float[Array, ""] = 1.0,
    dt0: float | Float[Array, ""] = 0.1,
) -> PKPDState:
    """Integrate PK from ``t0`` to ``t1`` with Dopri5 (no Python time loop)."""
    term = diffrax.ODETerm(pk_vector_field)
    solver = diffrax.Dopri5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=state.as_vector(),
        args=params,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    y1 = sol.ys[-1]
    return PKPDState.from_vector(y1)


def simulate_weekly_dosing(
    params: PKPDParams,
    weekly_doses_mg: Float[Array, " n_weeks"],
    days_per_week: int = 7,
    dt0: float = 0.1,
) -> tuple[PKPDState, Float[Array, " n_weeks n_days 2"], Float[Array, " n_weeks n_days"]]:
    """Apply weekly boluses then scan daily integration (JIT / vmap compatible)."""

    def week_step(
        state: PKPDState, dose: Float[Array, ""]
    ) -> tuple[PKPDState, tuple[Float[Array, " n_days 2"], Float[Array, " n_days"]]]:
        state = inject_dose(state, dose, params)

        def day_step(
            carry: PKPDState, _: Any
        ) -> tuple[PKPDState, tuple[Float[Array, "2"], Float[Array, ""]]]:
            nxt = integrate_pk(carry, params, t0=0.0, t1=1.0, dt0=dt0)
            effect = emax_pd(nxt.concentration, params)
            return nxt, (nxt.as_vector(), effect)

        state, (traj, effects) = jax.lax.scan(
            day_step, state, None, length=days_per_week
        )
        return state, (traj, effects)

    final_state, (trajectories, pd_effects) = jax.lax.scan(
        week_step, PKPDState.zeros(), weekly_doses_mg
    )
    return final_state, trajectories, pd_effects


if __name__ == "__main__":
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    del key  # PK/PD vector field is deterministic; key reserved for env noise.

    params = PKPDParams.semaglutide()
    state0 = PKPDState.zeros()
    dose = jnp.asarray(0.25)

    integrate_jit = jax.jit(integrate_pk)
    state1 = integrate_jit(inject_dose(state0, dose, params), params)

    assert state1.depot.shape == ()
    assert state1.concentration.shape == ()
    assert jnp.isfinite(state1.depot)
    assert jnp.isfinite(state1.concentration)
    assert state1.concentration > 0.0

    weekly_doses = jnp.full((4,), 0.25)
    simulate_jit = jax.jit(simulate_weekly_dosing)
    final, traj, effects = simulate_jit(params, weekly_doses)

    assert traj.shape == (4, 7, 2)
    assert effects.shape == (4, 7)
    assert final.as_vector().shape == (2,)
    assert jnp.all(jnp.isfinite(traj))
    assert jnp.all(effects >= 0.0) and jnp.all(effects <= params.e_max + 1e-6)

    batched_doses = jnp.stack([weekly_doses, weekly_doses * 2.0])
    _, traj_b, _ = jax.vmap(simulate_weekly_dosing, in_axes=(None, 0))(
        params, batched_doses
    )
    assert traj_b.shape == (2, 4, 7, 2)

    print("pkpd sanity: jit, shapes, and vmap ok")
    print(f"  C(t=1d | 0.25 mg) = {float(state1.concentration):.6f} mg/L")
    print(f"  C(week 4, day 7)  = {float(final.concentration):.6f} mg/L")
    print(f"  PD effect (last)  = {float(effects[-1, -1]):.4f}")
