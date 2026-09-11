"""Objective-evaluation accounting, budgets, traces, and target tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


class BudgetExhausted(RuntimeError):
    """Raised before an objective call that would exceed the NFE budget."""


@dataclass
class EvaluationTrace:
    """Best-so-far error sampled on a fixed grid of function evaluations."""

    grid: np.ndarray
    values: np.ndarray

    def as_dict(self) -> dict:
        return {"nfe": self.grid.tolist(), "best_error": self.values.tolist()}


@dataclass
class CountingObjective:
    """Counts, budgets, and traces every objective evaluation.

    Parameters
    ----------
    objective:
        The raw objective, called with a 1-D array.
    budget:
        Maximum number of objective calls. ``None`` disables the limit.
    optimum:
        Known ``f*``; the trace stores ``f - f*`` so errors are comparable
        across functions with different biases.
    trace_grid:
        FE checkpoints at which the best-so-far error is recorded.
    error_floor:
        Errors below this are clamped to 0, matching the CEC 2017 convention
        that differences below 1e-8 are reported as zero.
    """

    objective: Callable[[np.ndarray], float]
    budget: Optional[int] = None
    optimum: float = 0.0
    trace_grid: Optional[Sequence[int]] = None
    error_floor: float = 1e-8

    n_evaluations: int = field(default=0, init=False)
    best_value: float = field(default=float("inf"), init=False)
    best_position: Optional[np.ndarray] = field(default=None, init=False)
    _grid: np.ndarray = field(init=False, repr=False)
    _trace: List[float] = field(default_factory=list, init=False, repr=False)
    _grid_cursor: int = field(default=0, init=False, repr=False)
    _target_hits: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trace_grid is None:
            self._grid = np.asarray([], dtype=np.int64)
        else:
            self._grid = np.asarray(sorted(set(int(v) for v in self.trace_grid)), dtype=np.int64)
        self._trace = []

    # -- core ---------------------------------------------------------- #
    def __call__(self, x) -> float:
        if self.budget is not None and self.n_evaluations >= self.budget:
            raise BudgetExhausted(
                f"objective budget of {self.budget} evaluations is exhausted"
            )
        vector = np.asarray(x, dtype=float).reshape(-1)
        value = float(self.objective(vector))
        self.n_evaluations += 1
        if value < self.best_value:
            self.best_value = value
            self.best_position = vector.copy()
        self._record_checkpoints()
        return value

    @property
    def best_error(self) -> float:
        if not np.isfinite(self.best_value):
            return float("inf")
        error = self.best_value - self.optimum
        return 0.0 if error < self.error_floor else float(error)

    @property
    def remaining(self) -> Optional[int]:
        return None if self.budget is None else max(self.budget - self.n_evaluations, 0)

    def can_afford(self, count: int) -> bool:
        return self.budget is None or self.n_evaluations + int(count) <= self.budget

    # -- tracing ------------------------------------------------------- #
    def _record_checkpoints(self) -> None:
        while self._grid_cursor < self._grid.size and self.n_evaluations >= self._grid[self._grid_cursor]:
            self._trace.append(self.best_error)
            self._grid_cursor += 1

    def register_targets(self, targets: Sequence[float]) -> None:
        for target in targets:
            self._target_hits.setdefault(float(target), None)

    def update_targets(self) -> None:
        error = self.best_error
        for target in list(self._target_hits):
            if self._target_hits[target] is None and error <= target:
                self._target_hits[target] = int(self.n_evaluations)

    @property
    def target_hits(self) -> dict:
        return dict(self._target_hits)

    def finalize(self) -> EvaluationTrace:
        """Pad the trace to the full grid using the final best-so-far value."""
        while self._grid_cursor < self._grid.size:
            self._trace.append(self.best_error)
            self._grid_cursor += 1
        return EvaluationTrace(self._grid.copy(), np.asarray(self._trace, dtype=float))


def geometric_trace_grid(budget: int, n_points: int = 60) -> np.ndarray:
    """Log-spaced FE checkpoints, always including 1 and the full budget.

    A log grid resolves the early descent, where the algorithms differ most,
    without storing one row per evaluation.
    """
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    if n_points < 2:
        return np.asarray([1, budget], dtype=np.int64)
    raw = np.unique(np.round(np.geomspace(1, budget, num=n_points)).astype(np.int64))
    raw = raw[(raw >= 1) & (raw <= budget)]
    if raw[0] != 1:
        raw = np.concatenate(([1], raw))
    if raw[-1] != budget:
        raw = np.concatenate((raw, [budget]))
    return np.unique(raw)


DEFAULT_TARGETS = (1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)
