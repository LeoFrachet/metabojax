"""Unit tests for semaglutide 1-compartment PK/PD (Diffrax)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from metabojax.pkpd import (
    PKPDParams,
    PKPDState,
    emax_pd,
    inject_dose,
    integrate_pk,
    pk_vector_field,
    simulate_weekly_dosing,
)

T_HALF_DAYS = 7.0


@pytest.fixture
def params() -> PKPDParams:
    return PKPDParams.semaglutide()


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


def bateman_concentration(
    t: jax.Array,
    dose_mg: jax.Array,
    params: PKPDParams,
) -> jax.Array:
    """Closed-form plasma C(t) after a single SC bolus from rest."""
    absorbed = params.bioavailability * dose_mg
    ka, ke, vd = params.k_a, params.k_e, params.v_d
    return (absorbed / vd) * (ka / (ka - ke)) * (jnp.exp(-ke * t) - jnp.exp(-ka * t))


def test_elimination_rate_matches_seven_day_half_life(params: PKPDParams) -> None:
    assert params.k_e == pytest.approx(float(jnp.log(2.0) / T_HALF_DAYS))


def test_inject_dose_adds_bioavailable_amount_only_to_depot(params: PKPDParams) -> None:
    state0 = PKPDState.zeros()
    dose = jnp.asarray(0.25)
    state1 = inject_dose(state0, dose, params)

    assert float(state1.depot) == pytest.approx(float(params.bioavailability * dose))
    assert float(state1.concentration) == pytest.approx(0.0)
    assert float(state0.depot) == pytest.approx(0.0)


def test_inject_dose_is_pure(params: PKPDParams) -> None:
    state = PKPDState(depot=jnp.asarray(1.0), concentration=jnp.asarray(0.01))
    before = state.as_vector()
    _ = inject_dose(state, jnp.asarray(0.5), params)
    assert jnp.array_equal(state.as_vector(), before)


def test_vector_field_at_rest_is_zero(params: PKPDParams) -> None:
    dy = pk_vector_field(jnp.asarray(0.0), jnp.zeros(2), params)
    assert jnp.allclose(dy, jnp.zeros(2))


def test_vector_field_depot_decays_and_feeds_plasma(params: PKPDParams) -> None:
    y = jnp.asarray([1.0, 0.0])
    dy = pk_vector_field(jnp.asarray(0.0), y, params)
    assert float(dy[0]) < 0.0
    assert float(dy[1]) > 0.0
    assert float(dy[0]) == pytest.approx(float(-params.k_a * y[0]))
    assert float(dy[1]) == pytest.approx(float(params.k_a * y[0] / params.v_d))


def test_emax_pd_limits(params: PKPDParams) -> None:
    assert float(emax_pd(jnp.asarray(0.0), params)) == pytest.approx(0.0)
    half = emax_pd(params.e_c50, params)
    assert float(half) == pytest.approx(float(0.5 * params.e_max), rel=1e-5)
    sat = emax_pd(jnp.asarray(1e6), params)
    assert float(sat) == pytest.approx(float(params.e_max), rel=1e-5)


def test_integrate_pk_matches_bateman_solution(params: PKPDParams) -> None:
    dose = jnp.asarray(0.25)
    t1 = 1.0
    state = inject_dose(PKPDState.zeros(), dose, params)
    nxt = integrate_pk(state, params, t0=0.0, t1=t1)

    absorbed = params.bioavailability * dose
    depot_exact = absorbed * jnp.exp(-params.k_a * t1)
    c_exact = bateman_concentration(jnp.asarray(t1), dose, params)

    assert float(nxt.depot) == pytest.approx(float(depot_exact), rel=1e-4)
    assert float(nxt.concentration) == pytest.approx(float(c_exact), rel=1e-3)
    assert float(nxt.concentration) > 0.0


def test_integrate_pk_jit_and_shapes(params: PKPDParams, key: jax.Array) -> None:
    del key
    state = inject_dose(PKPDState.zeros(), jnp.asarray(0.25), params)
    nxt = jax.jit(integrate_pk)(state, params)
    assert nxt.depot.shape == ()
    assert nxt.concentration.shape == ()
    assert jnp.isfinite(nxt.depot)
    assert jnp.isfinite(nxt.concentration)


def test_simulate_weekly_dosing_shapes_and_pd_bounds(params: PKPDParams) -> None:
    weekly_doses = jnp.full((4,), 0.25)
    final, traj, effects = jax.jit(simulate_weekly_dosing)(params, weekly_doses)

    assert traj.shape == (4, 7, 2)
    assert effects.shape == (4, 7)
    assert final.as_vector().shape == (2,)
    assert bool(jnp.all(jnp.isfinite(traj)))
    assert bool(jnp.all(effects >= 0.0))
    assert bool(jnp.all(effects <= params.e_max + 1e-6))
    assert float(final.concentration) > 0.0


def test_higher_weekly_dose_raises_concentration(params: PKPDParams) -> None:
    low = jnp.full((4,), 0.25)
    high = jnp.full((4,), 0.50)
    simulate = jax.jit(simulate_weekly_dosing)
    _, traj_low, _ = simulate(params, low)
    _, traj_high, _ = simulate(params, high)
    assert float(traj_high[-1, -1, 1]) > float(traj_low[-1, -1, 1])


def test_simulate_weekly_dosing_vmap(params: PKPDParams) -> None:
    weekly = jnp.full((4,), 0.25)
    batched = jnp.stack([weekly, weekly * 2.0])
    _, traj_b, effects_b = jax.vmap(simulate_weekly_dosing, in_axes=(None, 0))(
        params, batched
    )
    assert traj_b.shape == (2, 4, 7, 2)
    assert effects_b.shape == (2, 4, 7)
    assert float(traj_b[1, -1, -1, 1]) > float(traj_b[0, -1, -1, 1])
