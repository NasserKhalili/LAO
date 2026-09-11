"""Tests for the LAO selection mechanism and supporting components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lao import COMPONENTS, LAO, CountingObjective  # noqa: E402
from lao.environment import (  # noqa: E402
    CONTINUOUS_TURBULENCE,
    DISCRETE_GUST,
    LONG_TERM_DISTRIBUTION,
    MEAN_PROFILE,
    PROCESS_ROLES,
    EnvironmentField,
)
from lao.leaf import (  # noqa: E402
    LOGISTIC_P3_FACTOR,
    SPECIES_PARAMS,
    Leaf,
    cumulative_leaf_fall,
    phenology_landmarks,
)

DIM = 8
BOUNDS = (np.full(DIM, -10.0), np.full(DIM, 10.0))


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def _optimizer(**overrides):
    budget = overrides.pop("budget", 1200)
    params = {"max_evaluations": budget, "seed": 4, "diagnostics": True}
    params.update(overrides)
    counter = CountingObjective(sphere, budget=budget, optimum=0.0)
    return LAO(counter, DIM, BOUNDS, params=params)


# Phenology

def test_logistic_p3_encodes_the_ten_percent_point():
    """At t = P2 - P3 the Wang et al. logistic must give 10 % cumulative fall."""
    for params in SPECIES_PARAMS.values():
        peak = float(params["peak"])
        spread = float(params["spread"])
        value = cumulative_leaf_fall(peak - spread, peak, spread)
        assert value == pytest.approx(0.1, abs=2e-3)
        assert cumulative_leaf_fall(peak, peak, spread) == pytest.approx(0.5)
    assert LOGISTIC_P3_FACTOR == pytest.approx(2.2)


def test_phenology_landmarks_are_ordered_and_consistent():
    for params in SPECIES_PARAMS.values():
        marks = phenology_landmarks(float(params["peak"]), float(params["spread"]))
        assert marks["start_10pct"] < marks["peak_50pct"] < marks["end_90pct"]
        assert marks["duration_10_90"] == pytest.approx(
            marks["end_90pct"] - marks["start_10pct"], rel=1e-9
        )


def test_species_strength_ordering_matches_the_reported_classes():
    strengths = [float(SPECIES_PARAMS[k]["strength"]) for k in sorted(SPECIES_PARAMS)]
    peaks = [float(SPECIES_PARAMS[k]["peak"]) for k in sorted(SPECIES_PARAMS)]
    assert strengths == sorted(strengths)   # early < intermediate < late
    assert peaks == sorted(peaks)


# Environmental field

def test_environment_roles_are_not_interchangeable():
    assert PROCESS_ROLES["Dryden"] == CONTINUOUS_TURBULENCE
    assert PROCESS_ROLES["VonKarman"] == CONTINUOUS_TURBULENCE
    assert PROCESS_ROLES["OneCosine"] == DISCRETE_GUST
    assert PROCESS_ROLES["PowerLaw"] == MEAN_PROFILE
    assert PROCESS_ROLES["Logarithmic"] == MEAN_PROFILE
    assert PROCESS_ROLES["Weibull"] == LONG_TERM_DISTRIBUTION


def test_field_state_is_independent_of_population_size():
    """The field must not depend on how many candidates query it."""
    states = {}
    for population in (5, 50):
        field = EnvironmentField()
        rng = np.random.default_rng(0)
        trace = []
        for step in range(25):
            trace.append(field.step(step / 25.0, rng).continuous)
            field.exposure(np.linspace(0.2, 1.0, population), np.random.default_rng(1))
        states[population] = trace
    np.testing.assert_allclose(states[5], states[50], rtol=0, atol=0)


def test_exposure_is_monotone_in_height_without_noise():
    field = EnvironmentField(exposure_noise=0.0)
    rng = np.random.default_rng(2)
    for _ in range(5):
        field.step(0.5, rng)
    heights = np.array([0.2, 0.5, 0.9, 1.0])
    exposure = field.exposure(heights, rng)
    assert np.all(np.diff(exposure) >= -1e-12)


# Threshold selection

def test_strength_is_rank_based_and_scale_invariant():
    """LS must be invariant to any strictly increasing rescaling of fitness."""
    optimizer = _optimizer()
    optimizer._initialise()
    baseline = optimizer._intrinsic_strength().copy()
    for leaf in optimizer.leaves:
        leaf.fitness = float(np.log1p(leaf.fitness) * 1e6 + 7.0)
    rescaled = optimizer._intrinsic_strength()
    np.testing.assert_allclose(baseline, rescaled, rtol=0, atol=0)


def test_best_leaf_has_maximum_strength_and_worst_has_zero():
    optimizer = _optimizer()
    optimizer._initialise()
    strength = optimizer._intrinsic_strength()
    order = np.argsort([leaf.fitness for leaf in optimizer.leaves])
    assert strength[order[-1]] == pytest.approx(0.0)
    assert strength[order[0]] > 0.0
    assert np.all(strength >= 0.0) and np.all(strength <= 1.0)


def test_pressure_is_bounded_and_increases_when_diversity_collapses():
    optimizer = _optimizer()
    optimizer._initialise()
    optimizer._intrinsic_strength()
    exposure = np.zeros(len(optimizer.leaves))
    rich = optimizer._abscission_pressure(0.5, diversity=1.0, exposure=exposure)
    poor = optimizer._abscission_pressure(0.5, diversity=0.0, exposure=exposure)
    assert np.all((rich >= 0.0) & (rich <= 1.0))
    assert np.mean(poor) >= np.mean(rich)


def test_only_threshold_crossing_slots_are_replaced():
    """Slots whose pressure does not exceed their strength stay bit-identical."""
    optimizer = _optimizer(budget=900)
    optimizer._initialise()
    tau = 0.4
    field = optimizer.environment.step(tau, optimizer.rng).continuous
    heights = np.asarray([leaf.height for leaf in optimizer.leaves])
    exposure = optimizer.environment.exposure(heights, optimizer.rng)
    diversity = optimizer._diversity()
    strength = optimizer._intrinsic_strength()
    pressure = optimizer._abscission_pressure(tau, diversity, exposure)
    crossing = set(np.flatnonzero(pressure > strength).tolist())
    before = [leaf.position.copy() for leaf in optimizer.leaves]
    for index in sorted(crossing):
        optimizer._replace_slot(optimizer.leaves[index], tau, field)
    for index, leaf in enumerate(optimizer.leaves):
        if index in crossing:
            continue
        np.testing.assert_array_equal(leaf.position, before[index])


def test_elite_refinement_never_degrades_a_slot():
    optimizer = _optimizer(budget=1500, enable_elite=True)
    optimizer._initialise()
    before = [leaf.fitness for leaf in optimizer.leaves]
    trials, accepted = optimizer._refine_elites(0.2)
    assert trials > 0 and accepted <= trials
    for index, leaf in enumerate(optimizer.leaves):
        assert leaf.fitness <= before[index] + 1e-15


def test_leaf_replace_is_the_only_mutation_path():
    leaf = Leaf(position=np.zeros(3))
    assert leaf.replacements == 0
    leaf.replace(np.ones(3), 1.0)
    assert leaf.replacements == 1
    np.testing.assert_array_equal(leaf.position, np.ones(3))


# Vibration schedule

def test_shipped_defaults_are_the_selected_architecture():
    """The default configuration corresponds to LAO-Core."""
    from lao.lao import DEFAULT_PARAMS

    for component in COMPONENTS:
        assert DEFAULT_PARAMS[f"enable_{component}"] is False, (
            f"'{component}' is enabled by default, but the blocked factorial ablation "
            "did not establish a benefit for it"
        )
    optimizer = _optimizer(budget=1200)
    assert all(value is False for value in optimizer.describe()["components"].values())


def test_vibration_schedule_is_in_normalised_time_and_fires():
    optimizer = _optimizer(enable_vibration=True, vibration_events=4, budget=4000)
    assert optimizer._vibration_times.size == 4
    assert np.all((optimizer._vibration_times > 0.0) & (optimizer._vibration_times < 1.0))
    result = optimizer.optimize()
    fired = result.diagnostics["vibration_fired"].sum()
    assert fired == 4, "every scheduled disturbance must fire within one run"


def test_vibration_can_be_disabled():
    optimizer = _optimizer(enable_vibration=False, budget=1500)
    assert optimizer._vibration_times.size == 0
    result = optimizer.optimize()
    assert result.diagnostics["vibration_fired"].sum() == 0


# Component switches

@pytest.mark.parametrize("component", COMPONENTS)
def test_each_component_switch_changes_the_trajectory(component):
    """Each optional component can alter the search trajectory when enabled."""
    budget = 2500
    all_on = {f"enable_{c}": True for c in COMPONENTS}
    with_component = _optimizer(budget=budget, **all_on).optimize()
    without = _optimizer(budget=budget, **{**all_on, f"enable_{component}": False}).optimize()
    assert with_component.best_position.tolist() != without.best_position.tolist(), (
        f"disabling '{component}' left the run unchanged, so the component is inert"
    )


def test_describe_reports_components_and_environment():
    description = _optimizer(**{f"enable_{c}": True for c in COMPONENTS}).describe()
    assert set(description["components"]) == set(COMPONENTS)
    assert description["environment"]["forcing_role"] == CONTINUOUS_TURBULENCE
    assert description["paradigm"] == "threshold-based selection"


def test_threshold_guard_respects_terminal_budget():
    optimizer = _optimizer(budget=31)
    result = optimizer.optimize()
    assert result.n_evaluations == 31
    assert result.n_evaluations <= 31
    assert result.stopped_by_budget


def test_diagnostics_have_consistent_lengths():
    result = _optimizer(budget=3000).optimize()
    lengths = {key: value.size for key, value in result.diagnostics.items()}
    assert len(set(lengths.values())) == 1, lengths
    assert result.diagnostics["replacement_rate"].max() <= 1.0 + 1e-12
