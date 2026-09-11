"""Baseline optimizers under a common objective-evaluation protocol.

Every optimizer here shares three properties that the comparison depends on:

1. It never calls the raw objective.  It calls a
   :class:`~lao.accounting.CountingObjective`, so the reported
   ``n_evaluations`` and the enforced budget come from the same counter for
   every algorithm, and no algorithm can overspend.
2. Any schedule that a canonical description writes as a function of the
   iteration index (the GWO/WOA coefficient ``a``, a decaying inertia weight)
   is evaluated as a function of *consumed evaluations*.  Under an equal-FE
   protocol, iteration-indexed schedules advance at different real rates for
   algorithms with different per-iteration costs, which would silently favour
   whichever algorithm has the cheapest iteration.
3. Parameters are the published defaults for that algorithm and are recorded
   through the experiment configuration.  Nothing is re-tuned by the optimizer implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .accounting import BudgetExhausted, CountingObjective


@dataclass
class OptimizerResult:
    best_position: np.ndarray
    best_fitness: float
    best_error: float
    n_evaluations: int
    iterations_run: int
    runtime_seconds: float
    stopped_by_budget: bool
    success: bool = True
    message: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    trace_nfe: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    trace_error: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    target_hits: Dict[float, Optional[int]] = field(default_factory=dict)
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)


class _Base:
    """Shared budget, counter, and result plumbing."""

    name = "base"
    defaults: Dict[str, Any] = {}

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dim = int(dim)
        self.lb = np.asarray(bounds[0], dtype=float)
        self.ub = np.asarray(bounds[1], dtype=float)
        if self.lb.shape != (dim,) or self.ub.shape != (dim,):
            raise ValueError("bounds must have shape (dim,) on each side")
        self.span = self.ub - self.lb

        merged: Dict[str, Any] = {"max_evaluations": None, "seed": None}
        merged.update(self.defaults)
        if params:
            unknown = sorted(set(params) - set(merged))
            if unknown:
                raise ValueError(f"unknown {self.name} parameter(s): {', '.join(unknown)}")
            merged.update(params)
        self.params = merged
        self.params = self._resolve(self.params)

        budget = self.params.get("max_evaluations")
        budget = None if budget is None else int(budget)
        if isinstance(objective, CountingObjective):
            self.counter = objective
            if budget is not None and self.counter.budget is None:
                self.counter.budget = budget
        else:
            self.counter = CountingObjective(objective, budget=budget)
        self.budget = self.counter.budget
        self.rng = np.random.default_rng(self.params["seed"])
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness = float("inf")

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for dimension-dependent defaults."""
        return params

    # -- helpers -------------------------------------------------------- #
    @property
    def fe_progress(self) -> float:
        if self.budget is None:
            return 0.0
        return float(np.clip(self.counter.n_evaluations / self.budget, 0.0, 1.0))

    def _evaluate(self, x: np.ndarray) -> float:
        value = self.counter(x)
        if value < self.best_fitness:
            self.best_fitness = value
            self.best_position = np.asarray(x, dtype=float).copy()
        return value

    def _uniform(self, n: int) -> np.ndarray:
        return self.lb + self.rng.random((n, self.dim)) * self.span

    def _finish(self, start: float, iterations: int, stopped: bool) -> OptimizerResult:
        trace = self.counter.finalize()
        self.counter.update_targets()
        if self.best_position is None:
            self.best_position = self.counter.best_position
            self.best_fitness = self.counter.best_value
        return OptimizerResult(
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
            best_error=self.counter.best_error,
            n_evaluations=int(self.counter.n_evaluations),
            iterations_run=int(iterations),
            runtime_seconds=time.perf_counter() - start,
            stopped_by_budget=stopped,
            params=dict(self.params),
            trace_nfe=trace.grid,
            trace_error=trace.values,
            target_hits=self.counter.target_hits,
            message="evaluation budget exhausted" if stopped else "",
        )

    def describe(self) -> Dict[str, Any]:
        return {"algorithm": self.name, "parameters": dict(self.params)}

    def optimize(self) -> OptimizerResult:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------- #
# PSO
# --------------------------------------------------------------------- #
class PSO(_Base):
    """PSO with the Clerc & Kennedy (2002) constriction parameterisation."""

    name = "PSO"
    defaults = {
        "population_size": 30,
        "inertia": 0.7298,
        "cognitive": 1.49618,
        "social": 1.49618,
        "velocity_clamp": 0.2,
    }

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n = int(self.params["population_size"])
        w = float(self.params["inertia"])
        c1 = float(self.params["cognitive"])
        c2 = float(self.params["social"])
        v_max = float(self.params["velocity_clamp"]) * self.span
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n)
            V = self.rng.uniform(-1.0, 1.0, (n, self.dim)) * v_max
            F = np.array([self._evaluate(x) for x in X])
            P, Pf = X.copy(), F.copy()
            g = int(np.argmin(Pf))
            G, Gf = P[g].copy(), float(Pf[g])
            while True:
                iterations += 1
                r1 = self.rng.random((n, self.dim))
                r2 = self.rng.random((n, self.dim))
                V = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
                V = np.clip(V, -v_max, v_max)
                X = np.clip(X + V, self.lb, self.ub)
                for i in range(n):
                    F[i] = self._evaluate(X[i])
                    if F[i] < Pf[i]:
                        P[i] = X[i].copy()
                        Pf[i] = F[i]
                        if F[i] < Gf:
                            Gf = float(F[i])
                            G = X[i].copy()
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# GA
# --------------------------------------------------------------------- #
class GA(_Base):
    """Real-coded GA: tournament selection, SBX, polynomial mutation, elitism.

    SBX and polynomial mutation are applied with fully vectorised numpy
    expressions.  The distributions are exactly those of Deb & Agrawal (1995)
    and Deb & Goyal (1996); vectorising only removes the per-gene Python loop
    computationally infeasible.
    """

    name = "GA"
    defaults = {
        "population_size": 30,
        "tournament_size": 3,
        "crossover_rate": 0.9,
        "eta_crossover": 20.0,
        "mutation_rate": None,
        "eta_mutation": 20.0,
        "elite_count": 2,
    }

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("mutation_rate") is None:
            params["mutation_rate"] = 1.0 / self.dim
        return params

    def _tournament(self, F: np.ndarray, count: int) -> np.ndarray:
        k = int(self.params["tournament_size"])
        picks = self.rng.integers(0, F.size, size=(count, k))
        return picks[np.arange(count), np.argmin(F[picks], axis=1)]

    def _sbx(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eta = float(self.params["eta_crossover"])
        pairs, dim = p1.shape
        u = self.rng.random((pairs, dim))
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)),
            (1.0 / (2.0 * (1.0 - np.clip(u, None, 1.0 - 1e-15)))) ** (1.0 / (eta + 1.0)),
        )
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        # Genes that coincide, and pairs that fail the crossover trial, are
        # inherited unchanged, exactly as in the scalar formulation.
        identical = np.abs(p1 - p2) < 1e-14
        c1 = np.where(identical, p1, c1)
        c2 = np.where(identical, p2, c2)
        do_cross = self.rng.random((pairs, 1)) <= float(self.params["crossover_rate"])
        c1 = np.where(do_cross, c1, p1)
        c2 = np.where(do_cross, c2, p2)
        return np.clip(c1, self.lb, self.ub), np.clip(c2, self.lb, self.ub)

    def _mutate(self, X: np.ndarray) -> np.ndarray:
        eta = float(self.params["eta_mutation"])
        rate = float(self.params["mutation_rate"])
        u = self.rng.random(X.shape)
        delta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - np.clip(u, None, 1.0 - 1e-15))) ** (1.0 / (eta + 1.0)),
        )
        mask = self.rng.random(X.shape) < rate
        mutated = np.where(mask, X + delta * self.span, X)
        return np.clip(mutated, self.lb, self.ub)

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n = int(self.params["population_size"])
        elites = int(self.params["elite_count"])
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n)
            F = np.array([self._evaluate(x) for x in X])
            while True:
                iterations += 1
                order = np.argsort(F, kind="stable")
                offspring = [X[order[i]].copy() for i in range(min(elites, n))]
                needed = n - len(offspring)
                pairs = (needed + 1) // 2
                i1 = self._tournament(F, pairs)
                i2 = self._tournament(F, pairs)
                c1, c2 = self._sbx(X[i1], X[i2])
                children = np.concatenate((c1, c2), axis=0)[:needed]
                children = self._mutate(children)
                X = np.concatenate((np.asarray(offspring), children), axis=0)
                for i in range(n):
                    F[i] = self._evaluate(X[i])
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# DE
# --------------------------------------------------------------------- #
class DE(_Base):
    """Canonical DE/rand/1/bin with fixed F and CR (Storn & Price, 1997)."""

    name = "DE"
    defaults = {"population_size": 30, "differential_weight": 0.5, "crossover_rate": 0.9}

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n = int(self.params["population_size"])
        F_scale = float(self.params["differential_weight"])
        cr = float(self.params["crossover_rate"])
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n)
            fitness = np.array([self._evaluate(x) for x in X])
            while True:
                iterations += 1
                for i in range(n):
                    choices = self.rng.choice(n - 1, size=3, replace=False)
                    choices = np.where(choices >= i, choices + 1, choices)
                    r1, r2, r3 = (int(c) for c in choices)
                    mutant = np.clip(X[r1] + F_scale * (X[r2] - X[r3]), self.lb, self.ub)
                    cross = self.rng.random(self.dim) < cr
                    cross[self.rng.integers(0, self.dim)] = True
                    trial = np.where(cross, mutant, X[i])
                    value = self._evaluate(trial)
                    if value <= fitness[i]:
                        X[i] = trial
                        fitness[i] = value
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# GWO
# --------------------------------------------------------------------- #
class GWO(_Base):
    """Grey Wolf Optimizer with an FE-indexed ``a`` schedule."""

    name = "GWO"
    defaults = {"population_size": 30}

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n = int(self.params["population_size"])
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n)
            F = np.array([self._evaluate(x) for x in X])
            order = np.argsort(F, kind="stable")
            leaders = [X[order[j]].copy() for j in range(3)]
            leader_fitness = [float(F[order[j]]) for j in range(3)]
            while True:
                iterations += 1
                a = 2.0 - 2.0 * self.fe_progress
                for i in range(n):
                    proposal = np.zeros(self.dim)
                    for leader in leaders:
                        A = 2.0 * a * self.rng.random(self.dim) - a
                        C = 2.0 * self.rng.random(self.dim)
                        proposal += leader - A * np.abs(C * leader - X[i])
                    X[i] = np.clip(proposal / 3.0, self.lb, self.ub)
                    F[i] = self._evaluate(X[i])
                    for rank in range(3):
                        if F[i] < leader_fitness[rank]:
                            leaders.insert(rank, X[i].copy())
                            leader_fitness.insert(rank, float(F[i]))
                            leaders.pop()
                            leader_fitness.pop()
                            break
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# WOA
# --------------------------------------------------------------------- #
class WOA(_Base):
    """Whale Optimization Algorithm with an FE-indexed ``a`` schedule."""

    name = "WOA"
    defaults = {"population_size": 30, "spiral_constant": 1.0}

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n = int(self.params["population_size"])
        b = float(self.params["spiral_constant"])
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n)
            F = np.array([self._evaluate(x) for x in X])
            g = int(np.argmin(F))
            G, Gf = X[g].copy(), float(F[g])
            while True:
                iterations += 1
                a = 2.0 - 2.0 * self.fe_progress
                for i in range(n):
                    A = 2.0 * a * self.rng.random(self.dim) - a
                    C = 2.0 * self.rng.random(self.dim)
                    if self.rng.random() < 0.5:
                        if np.linalg.norm(A) < 1.0:
                            X[i] = G - A * np.abs(C * G - X[i])
                        else:
                            other = X[self.rng.integers(0, n)]
                            X[i] = other - A * np.abs(C * other - X[i])
                    else:
                        L = self.rng.uniform(-1.0, 1.0, self.dim)
                        X[i] = np.abs(G - X[i]) * np.exp(b * L) * np.cos(2.0 * np.pi * L) + G
                    X[i] = np.clip(X[i], self.lb, self.ub)
                    F[i] = self._evaluate(X[i])
                    if F[i] < Gf:
                        Gf = float(F[i])
                        G = X[i].copy()
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# CMA-ES
# --------------------------------------------------------------------- #
class CMAES(_Base):
    """CMA-ES via ``pycma``, driven purely by the shared FE budget.

    ``pycma``'s internal termination tests are disabled so that the equal-FE
    protocol -- not an algorithm-specific stopping rule -- defines the end of
    the run.  The initial mean is a uniform draw from the box, matching the
    initialisation of every other optimizer here; using the box centre
    instead would give CMA-ES a prior that the shifted CEC 2017 optima do not
    justify.
    """

    name = "CMAES"
    defaults = {"population_size": None, "sigma0_fraction": 0.3}

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        try:
            import cma
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CMAES requires the optional 'cma' package") from exc

        sigma0 = float(self.params["sigma0_fraction"]) * float(np.mean(self.span))
        x0 = (self.lb + self.rng.random(self.dim) * self.span).tolist()
        seed = self.params.get("seed")
        options = {
            "bounds": [self.lb.tolist(), self.ub.tolist()],
            "seed": None if seed is None else int(seed) + 1,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "tolx": 0.0,
            "tolfun": 0.0,
            "tolfunhist": 0.0,
            "tolflatfitness": 10 ** 9,
            "tolstagnation": 10 ** 9,
            "tolconditioncov": False,
        }
        if self.params["population_size"]:
            options["popsize"] = int(self.params["population_size"])
        strategy = cma.CMAEvolutionStrategy(x0, sigma0, options)
        self.params = {**self.params, "effective_population_size": int(strategy.popsize)}

        stopped = False
        iterations = 0
        try:
            while True:
                iterations += 1
                candidates = strategy.ask()
                values = [self._evaluate(np.asarray(c, dtype=float)) for c in candidates]
                strategy.tell(candidates, values)
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


# --------------------------------------------------------------------- #
# SHADE and L-SHADE
# --------------------------------------------------------------------- #
def _weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    numerator = float(np.sum(weights * values * values))
    denominator = float(np.sum(weights * values))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total == 0.0:
        return 0.0
    return float(np.sum(weights * values) / total)


class SHADE(_Base):
    """Success-History based Adaptive DE (Tanabe & Fukunaga, 2013).

    Implements current-to-pbest/1/bin with an external archive of replaced
    parents, per-individual ``F`` and ``CR`` sampled from a success-history
    memory, and the midpoint bound-repair rule of the original paper.

    Budget-aware population size
    ----------------------------
    The published population sizes (100 for SHADE, ``18 D`` for L-SHADE) were
    chosen for ``MaxFEs = 10000 D``.  Under a smaller budget those defaults
    leave too few generations for the success-history memory to adapt at all,
    which would understate the baseline rather than test it.  The population is
    therefore capped so that at least ``min_generations`` generations fit in the
    budget, and both the published and the effective value are recorded in the
    run manifest.  Setting ``min_generations`` to 0 restores the published
    default exactly.
    """

    name = "SHADE"
    defaults = {
        "population_size": 100,
        "memory_size": 100,
        "archive_rate": 1.0,
        "p_best_min_count": 2,
        "p_best_max_rate": 0.2,
        "p_best_mode": "sampled",
        "initial_memory": 0.5,
        "population_min": 4,
        "linear_reduction": False,
        "min_generations": 50,
    }

    def _budget_cap(self, published: int) -> int:
        budget = self.params.get("max_evaluations")
        floor = int(self.params.get("population_min", 4))
        minimum_generations = int(self.params.get("min_generations", 0))
        if not budget or minimum_generations <= 0:
            return int(published)
        return int(max(floor, min(int(published), int(budget) // minimum_generations)))

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["published_population_size"] = int(params["population_size"])
        return params

    def _sample_control(self, memory_cr: np.ndarray, memory_f: np.ndarray,
                        index: int) -> Tuple[float, float]:
        mean_cr = memory_cr[index]
        if mean_cr < 0.0:  # terminal value, L-SHADE convention
            cr = 0.0
        else:
            cr = float(np.clip(self.rng.normal(mean_cr, 0.1), 0.0, 1.0))
        mean_f = memory_f[index]
        f = -1.0
        for _ in range(100):
            f = float(mean_f + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5)))
            if f > 0.0:
                break
        if f <= 0.0:
            f = 1e-3
        return cr, min(f, 1.0)

    def _repair(self, trial: np.ndarray, parent: np.ndarray) -> np.ndarray:
        low = trial < self.lb
        high = trial > self.ub
        if np.any(low):
            trial[low] = (self.lb[low] + parent[low]) / 2.0
        if np.any(high):
            trial[high] = (self.ub[high] + parent[high]) / 2.0
        return trial

    def _target_population(self, initial: int) -> int:
        if not self.params["linear_reduction"] or self.budget is None:
            return initial
        minimum = int(self.params["population_min"])
        fraction = self.counter.n_evaluations / self.budget
        return max(minimum, int(round(initial - (initial - minimum) * fraction)))

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        n_init = self._budget_cap(int(self.params["population_size"]))
        self.params = {**self.params, "effective_population_size": n_init}
        history = int(self.params["memory_size"])
        archive_capacity = max(1, int(round(float(self.params["archive_rate"]) * n_init)))
        p_min_count = int(self.params["p_best_min_count"])
        p_max_rate = float(self.params["p_best_max_rate"])
        p_sampled = str(self.params["p_best_mode"]) == "sampled"
        initial_memory = float(self.params["initial_memory"])

        memory_cr = np.full(history, initial_memory, dtype=float)
        memory_f = np.full(history, initial_memory, dtype=float)
        memory_index = 0
        archive: List[np.ndarray] = []
        stopped = False
        iterations = 0
        try:
            X = self._uniform(n_init)
            fitness = np.array([self._evaluate(x) for x in X])
            while True:
                iterations += 1
                n = X.shape[0]
                success_cr: List[float] = []
                success_f: List[float] = []
                improvements: List[float] = []
                order = np.argsort(fitness, kind="stable")
                for i in range(n):
                    slot = int(self.rng.integers(0, history))
                    cr, f = self._sample_control(memory_cr, memory_f, slot)
                    if p_sampled:
                        low = p_min_count / n
                        p_rate = self.rng.uniform(low, p_max_rate) if p_max_rate > low else low
                    else:
                        p_rate = p_max_rate
                    p_count = max(2, min(n, int(round(p_rate * n))))
                    pbest = X[order[int(self.rng.integers(0, p_count))]]

                    r1 = int(self.rng.integers(0, n))
                    guard = 0
                    while r1 == i and n > 1 and guard < 100:
                        r1 = int(self.rng.integers(0, n))
                        guard += 1
                    pool_size = n + len(archive)
                    r2 = int(self.rng.integers(0, pool_size))
                    guard = 0
                    while (r2 == i or r2 == r1) and pool_size > 2 and guard < 100:
                        r2 = int(self.rng.integers(0, pool_size))
                        guard += 1
                    x_r2 = X[r2] if r2 < n else archive[r2 - n]

                    mutant = X[i] + f * (pbest - X[i]) + f * (X[r1] - x_r2)
                    mutant = self._repair(mutant, X[i])
                    cross = self.rng.random(self.dim) < cr
                    cross[self.rng.integers(0, self.dim)] = True
                    trial = np.where(cross, mutant, X[i])
                    value = self._evaluate(trial)
                    if value <= fitness[i]:
                        if value < fitness[i]:
                            archive.append(X[i].copy())
                            success_cr.append(cr)
                            success_f.append(f)
                            improvements.append(abs(float(fitness[i]) - value))
                        X[i] = trial
                        fitness[i] = value

                while len(archive) > archive_capacity:
                    archive.pop(int(self.rng.integers(0, len(archive))))

                if success_cr:
                    weights = np.asarray(improvements, dtype=float)
                    if weights.sum() <= 0.0:
                        weights = np.ones_like(weights)
                    cr_values = np.asarray(success_cr, dtype=float)
                    f_values = np.asarray(success_f, dtype=float)
                    if memory_cr[memory_index] < 0.0 or np.max(cr_values) == 0.0:
                        memory_cr[memory_index] = -1.0
                    else:
                        memory_cr[memory_index] = _weighted_mean(cr_values, weights)
                    memory_f[memory_index] = _weighted_lehmer_mean(f_values, weights)
                    memory_index = (memory_index + 1) % history

                target = self._target_population(n_init)
                if target < X.shape[0]:
                    keep = np.argsort(fitness, kind="stable")[:target]
                    X = X[keep]
                    fitness = fitness[keep]
                    archive_capacity = max(1, int(round(float(self.params["archive_rate"]) * target)))
                    while len(archive) > archive_capacity:
                        archive.pop(int(self.rng.integers(0, len(archive))))
                self.counter.update_targets()
        except BudgetExhausted:
            stopped = True
        return self._finish(start, iterations, stopped)


class LSHADE(SHADE):
    """L-SHADE: SHADE with linear population size reduction (2014).

    ``population_size`` defaults to ``18 D`` and the archive rate to 2.6, the
    values published with the algorithm; the population then decreases
    linearly in consumed evaluations to a floor of 4.  Unlike SHADE, the
    greediness ``p`` is fixed at 0.11 rather than sampled per individual.
    """

    name = "LSHADE"
    defaults = {
        "population_size": None,      # resolved to 18 * D
        "population_factor": 18,
        "memory_size": 6,
        "archive_rate": 2.6,
        "p_best_min_count": 2,
        "p_best_max_rate": 0.11,
        "p_best_mode": "fixed",
        "initial_memory": 0.5,
        "population_min": 4,
        "linear_reduction": True,
        "min_generations": 50,
    }

    def _resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("population_size") is None:
            params["population_size"] = int(params["population_factor"]) * self.dim
        params["published_population_size"] = int(params["population_size"])
        return params


ALGORITHMS: Dict[str, type] = {
    "PSO": PSO,
    "GA": GA,
    "DE": DE,
    "GWO": GWO,
    "WOA": WOA,
    "CMAES": CMAES,
    "SHADE": SHADE,
    "LSHADE": LSHADE,
}
