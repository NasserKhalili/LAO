"""Tests for the evaluation counter, budget enforcement, and FE traces."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import cma  # noqa: F401
    HAVE_CMA = True
except ImportError:
    HAVE_CMA = False

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lao import (  # noqa: E402
    DEFAULT_TARGETS,
    OPTIMIZERS,
    BudgetExhausted,
    CountingObjective,
    geometric_trace_grid,
)

DIM = 6
BOUNDS = (np.full(DIM, -5.0), np.full(DIM, 5.0))


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_counter_counts_every_call_and_blocks_overspend():
    counter = CountingObjective(sphere, budget=5)
    for _ in range(5):
        counter(np.zeros(DIM))
    assert counter.n_evaluations == 5
    with pytest.raises(BudgetExhausted):
        counter(np.zeros(DIM))
    assert counter.n_evaluations == 5


def test_counter_tracks_best_and_error_floor():
    counter = CountingObjective(sphere, optimum=0.0, error_floor=1e-3)
    counter(np.full(DIM, 1.0))
    counter(np.full(DIM, 1e-4))
    assert counter.best_value == pytest.approx(DIM * 1e-8)
    assert counter.best_error == 0.0  # below the floor


def test_trace_grid_spans_the_budget():
    grid = geometric_trace_grid(1000, 40)
    assert grid[0] == 1
    assert grid[-1] == 1000
    assert np.all(np.diff(grid) > 0)


def test_trace_is_monotone_non_increasing_and_full_length():
    budget = 200
    grid = geometric_trace_grid(budget, 25)
    rng = np.random.default_rng(0)
    counter = CountingObjective(sphere, budget=budget, optimum=0.0, trace_grid=grid)
    for _ in range(budget):
        counter(rng.uniform(-5, 5, DIM))
    trace = counter.finalize()
    assert trace.grid.size == trace.values.size == grid.size
    assert np.all(np.diff(trace.values) <= 1e-12)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_optimizer_nfe_equals_counter_and_never_exceeds_budget(name):
    """The reported NFE must come from the objective-level counter."""
    budget = 240
    counter = CountingObjective(sphere, budget=budget, optimum=0.0,
                                trace_grid=geometric_trace_grid(budget, 20))
    counter.register_targets(DEFAULT_TARGETS)
    if name == "CMAES" and not HAVE_CMA:
        pytest.skip("cma is not installed")
    result = OPTIMIZERS[name](counter, DIM, BOUNDS,
                              params={"max_evaluations": budget, "seed": 3}).optimize()
    assert result.n_evaluations == counter.n_evaluations
    assert result.n_evaluations <= budget
    assert result.n_evaluations == budget
    assert np.isfinite(result.best_fitness)
    assert result.trace_error.size == result.trace_nfe.size


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_optimizer_is_deterministic_under_a_fixed_seed(name):
    budget = 200

    if name == "CMAES" and not HAVE_CMA:
        pytest.skip("cma is not installed")

    def run():
        counter = CountingObjective(sphere, budget=budget, optimum=0.0,
                                    trace_grid=geometric_trace_grid(budget, 15))
        return OPTIMIZERS[name](counter, DIM, BOUNDS,
                                params={"max_evaluations": budget, "seed": 11}).optimize()

    first, second = run(), run()
    assert first.best_fitness == second.best_fitness
    assert first.n_evaluations == second.n_evaluations
    np.testing.assert_allclose(first.trace_error, second.trace_error, rtol=0, atol=0)
    np.testing.assert_allclose(first.best_position, second.best_position, rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_different_seeds_give_different_trajectories(name):
    budget = 200

    if name == "CMAES" and not HAVE_CMA:
        pytest.skip("cma is not installed")

    def run(seed):
        counter = CountingObjective(sphere, budget=budget, optimum=0.0)
        return OPTIMIZERS[name](counter, DIM, BOUNDS,
                                params={"max_evaluations": budget, "seed": seed}).optimize()

    assert run(1).best_position.tolist() != run(2).best_position.tolist()


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_unknown_parameters_are_rejected(name):
    counter = CountingObjective(sphere, budget=100)
    with pytest.raises(ValueError):
        OPTIMIZERS[name](counter, DIM, BOUNDS,
                         params={"max_evaluations": 100, "seed": 0, "not_a_parameter": 1})


def test_target_hits_are_recorded_in_evaluations():
    budget = 400
    counter = CountingObjective(sphere, budget=budget, optimum=0.0)
    counter.register_targets([1e0, 1e-2])
    OPTIMIZERS["LSHADE"](counter, DIM, BOUNDS,
                         params={"max_evaluations": budget, "seed": 5}).optimize()
    hits = counter.target_hits
    assert set(hits) == {1e0, 1e-2}
    for target, nfe in hits.items():
        assert nfe is None or 1 <= nfe <= budget
