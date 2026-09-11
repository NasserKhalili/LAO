"""Leaf-Abscission Optimization (LAO) and threshold-based selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special

from .accounting import BudgetExhausted, CountingObjective
from .environment import EnvironmentField
from .leaf import (
    N_SPECIES,
    Leaf,
    species_seasonal_pressure,
    species_strength,
)

#: Optional component switches; the threshold core is always active.
COMPONENTS = ("drift", "gust", "vibration", "elite")

DEFAULT_PARAMS: Dict[str, Any] = {
    # -- population and horizon ------------------------------------------
    "population_size": 30,
    "max_evaluations": None,
    "max_iter": None,              # safety cap only; the budget is the horizon

    # -- threshold rule ---------------------------------------------------
    "seasonal_weight": 0.90,       # alpha
    "environmental_weight": 0.20,  # beta
    "diversity_weight": 0.50,      # lambda_div

    # -- regrowth kernel --------------------------------------------------
    "local_probability": 0.95,     # p_local
    "tournament_size": 3,
    "step_scale": 0.15,            # sigma_base
    "step_decay_exponent": 2.0,    # gamma
    "phototropism": 0.50,          # eta

    # -- component switches -----------------------------------------------
    "enable_drift": False,
    "enable_gust": False,
    "enable_vibration": False,
    "enable_elite": False,

    # -- continuous stochastic forcing (drift) ----------------------------
    "drift_scale": 0.02,           # c_drift, fraction of the box diagonal

    # -- discrete shock (gust) --------------------------------------------
    "gust_probability": 0.15,      # p_gust
    "gust_scale": 0.02,            # c_gust, fraction of the box diagonal
    "levy_alpha": 1.50,

    # -- controlled disturbance (vibration) -------------------------------
    "vibration_events": 6,         # events per run in normalised time
    "vibration_probability": 0.05,
    "vibration_strength_threshold": 0.15,

    # -- elite refinement -------------------------------------------------
    "elite_count": 5,
    "elite_trials": 3,
    "elite_step_init": 0.05,
    "elite_step_final": 1e-4,

    # -- environment ------------------------------------------------------
    "forcing_process": "Dryden",
    "vertical_profile": "PowerLaw",
    "exposure_noise": 0.05,
    "gust_arrival_rate": 0.05,
    "gust_duration": 0.02,
    "gust_step": 0.004,

    # -- bookkeeping ------------------------------------------------------
    "diagnostics": False,
    "seed": None,
    "verbose": False,
}


@dataclass
class LAOResult:
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
    #: Cumulative margin histograms over the whole run (20 bins on [-1, 1]):
    #: the denominator and numerator of P(replace | AP - LS). Not per-iteration,
    #: so they are held outside the diagnostics dict whose lengths must agree.
    margin_hist_total: np.ndarray = field(default_factory=lambda: np.zeros(0))
    margin_hist_replaced: np.ndarray = field(default_factory=lambda: np.zeros(0))


class LAO:
    """Threshold-based selection optimizer.

    Parameters
    ----------
    objective:
        Either a raw callable or a :class:`~lao.accounting.CountingObjective`.
        Passing the counter in is the supported path: it makes the reported
        ``n_evaluations`` come from the same object that gates the budget.
    """

    name = "LAO"

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
        self.span_norm = float(np.linalg.norm(self.span))

        self.params: Dict[str, Any] = {**DEFAULT_PARAMS}
        if params:
            unknown = sorted(set(params) - set(DEFAULT_PARAMS))
            if unknown:
                raise ValueError(f"unknown LAO parameter(s): {', '.join(unknown)}")
            self.params.update(params)

        population = int(self.params["population_size"])
        if population < 4:
            raise ValueError("population_size must be at least 4")
        budget = self.params.get("max_evaluations")
        if budget is not None:
            budget = int(budget)
            if budget < population:
                raise ValueError("max_evaluations must cover the initial population")
            self.params["max_evaluations"] = budget

        if isinstance(objective, CountingObjective):
            self.counter = objective
            if budget is not None and self.counter.budget is None:
                self.counter.budget = budget
        else:
            self.counter = CountingObjective(objective, budget=budget)
        self.budget = self.counter.budget

        self.rng = np.random.default_rng(self.params["seed"])
        self.environment = EnvironmentField(
            forcing=str(self.params["forcing_process"]),
            profile=str(self.params["vertical_profile"]),
            gust_duration=float(self.params["gust_duration"]),
            gust_arrival_rate=float(self.params["gust_arrival_rate"]),
            gust_step=float(self.params["gust_step"]),
            exposure_noise=float(self.params["exposure_noise"]),
            enable_gust=bool(self.params["enable_gust"]),
        )

        self.leaves: List[Leaf] = []
        self._raw_pressure_mean = 0.0
        self._raw_clip_above = 0.0
        self._margin_hist_total = np.zeros(20, dtype=float)
        self._margin_hist_replaced = np.zeros(20, dtype=float)
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf")
        self._vibration_times = self._vibration_schedule()
        self._vibration_cursor = 0
        self._levy_sigma = self._levy_scale(float(self.params["levy_alpha"]))

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _vibration_schedule(self) -> np.ndarray:
        """Normalised times at which the controlled disturbance fires.

        Events are placed at ``k / (n + 1)`` so they are interior to the run
        and evenly spaced in normalised optimisation time.  Expressing the
        schedule in normalised time (rather than as a fixed iteration period)
        is what makes the component behave comparably at D = 10, 30 and 50.
        """
        if not self.params["enable_vibration"]:
            return np.asarray([], dtype=float)
        n_events = int(self.params["vibration_events"])
        if n_events < 1:
            return np.asarray([], dtype=float)
        return (np.arange(1, n_events + 1, dtype=float)) / (n_events + 1.0)

    @staticmethod
    def _levy_scale(alpha: float) -> float:
        """Mantegna (1994) scale for the ``u`` component of a Levy sample."""
        numerator = special.gamma(1.0 + alpha) * np.sin(np.pi * alpha / 2.0)
        denominator = special.gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ** ((alpha - 1.0) / 2.0)
        return float((numerator / denominator) ** (1.0 / alpha))

    def _levy_sample(self, size: int) -> np.ndarray:
        alpha = float(self.params["levy_alpha"])
        u = self.rng.normal(0.0, self._levy_sigma, size=size)
        v = self.rng.normal(0.0, 1.0, size=size)
        v = np.where(np.abs(v) < 1e-12, 1e-12, v)
        return u / (np.abs(v) ** (1.0 / alpha))

    # ------------------------------------------------------------------ #
    # state descriptors
    # ------------------------------------------------------------------ #
    @property
    def progress(self) -> float:
        """Normalised optimisation time ``tau`` in [0, 1], indexed by FEs.

        Indexing progress by evaluations rather than iterations is required
        here: LAO's per-iteration evaluation count is variable, so an
        iteration-indexed schedule would advance at a data-dependent rate and
        would not be comparable with the fixed-cost baselines.
        """
        if self.budget is None:
            cap = self.params.get("max_iter") or 1
            return float(np.clip(self._iteration / cap, 0.0, 1.0))
        initial = int(self.params["population_size"])
        denominator = max(int(self.budget) - initial, 1)
        return float(np.clip((self.counter.n_evaluations - initial) / denominator, 0.0, 1.0))

    def _positions(self) -> np.ndarray:
        return np.asarray([leaf.position for leaf in self.leaves], dtype=float)

    def _diversity(self, positions: Optional[np.ndarray] = None) -> float:
        """Normalised mean pairwise distance of the population."""
        if len(self.leaves) < 2 or self.span_norm < 1e-12:
            return 0.0
        points = self._positions() if positions is None else positions
        n = points.shape[0]
        difference = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(difference, axis=-1)
        mean_distance = distance.sum() / (n * (n - 1))
        return float(mean_distance / self.span_norm)

    # ------------------------------------------------------------------ #
    # threshold rule
    # ------------------------------------------------------------------ #
    def _intrinsic_strength(self) -> np.ndarray:
        """``LS_i = ell(s_i) (1 - q_i)`` with ``q_i`` the normalised fitness rank.

        Rank-based strength depends on within-population ordering rather than
        the absolute scale of objective values. This makes the fitness-derived
        component invariant to strictly increasing transformations of the
        objective, including changes in numerical scale across benchmark
        functions.
        """
        fitness = np.asarray([leaf.fitness for leaf in self.leaves], dtype=float)
        n = fitness.size
        order = np.argsort(fitness, kind="stable")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        normalised_rank = rank / max(n - 1, 1)
        species = np.asarray([leaf.species for leaf in self.leaves], dtype=int)
        strength = species_strength(species) * (1.0 - normalised_rank)
        for index, leaf in enumerate(self.leaves):
            leaf.strength = float(strength[index])
        return strength

    def _abscission_pressure(self, tau: float, diversity: float,
                             exposure: np.ndarray) -> np.ndarray:
        """``AP_i = clip(alpha S_i(tau) B(div) + beta U_i xi_i, 0, 1)``."""
        alpha = float(self.params["seasonal_weight"])
        beta = float(self.params["environmental_weight"])
        lambda_div = float(self.params["diversity_weight"])
        species = np.asarray([leaf.species for leaf in self.leaves], dtype=int)
        seasonal = species_seasonal_pressure(tau, species)
        boost = 1.0 + lambda_div * (1.0 - diversity)
        susceptibility = self.rng.uniform(0.0, 1.0, size=species.size)
        raw = alpha * seasonal * boost + beta * exposure * susceptibility
        pressure = np.clip(raw, 0.0, 1.0)
        self._raw_pressure_mean = float(np.mean(raw)) if raw.size else 0.0
        self._raw_clip_above = float(np.mean(raw > 1.0)) if raw.size else 0.0
        for index, leaf in enumerate(self.leaves):
            leaf.pressure = float(pressure[index])
        return pressure

    # ------------------------------------------------------------------ #
    # regrowth
    # ------------------------------------------------------------------ #
    def _tournament_parent(self) -> Leaf:
        k = min(int(self.params["tournament_size"]), len(self.leaves))
        indices = self.rng.choice(len(self.leaves), size=k, replace=False)
        return min((self.leaves[i] for i in indices), key=lambda leaf: leaf.fitness)

    def _step_size(self, tau: float) -> float:
        base = float(self.params["step_scale"])
        gamma = float(self.params["step_decay_exponent"])
        return base * (1.0 - tau) ** gamma

    def _regrowth_candidate(self, tau: float, field_state: float,
                            force_local: bool) -> Tuple[np.ndarray, str]:
        """Propose a replacement position for one vacated slot."""
        if force_local or self.rng.random() < float(self.params["local_probability"]):
            parent = self._tournament_parent()
            candidate = parent.position.copy()
            if self.best_position is not None:
                candidate = candidate + float(self.params["phototropism"]) * (
                    self.best_position - parent.position
                )
            candidate = candidate + self.rng.normal(0.0, 1.0, self.dim) * self._step_size(tau) * self.span
            kind = "local"

            if self.params["enable_drift"] and field_state > 0.0:
                direction = self.rng.normal(0.0, 1.0, self.dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    candidate = candidate + (
                        float(self.params["drift_scale"]) * field_state
                        * (direction / norm) * self.span_norm
                    )
                    kind = "local+drift"

            if (self.params["enable_gust"]
                    and self.environment.state.gust > 0.0
                    and self.rng.random() < float(self.params["gust_probability"])):
                levy = self._levy_sample(self.dim)
                direction = self.rng.normal(0.0, 1.0, self.dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    candidate = candidate + (
                        float(self.params["gust_scale"]) * self.environment.state.gust
                        * levy * (direction / norm) * self.span_norm
                    )
                    kind = "local+gust"
        else:
            candidate = self.lb + self.rng.random(self.dim) * self.span
            kind = "global"
        return np.clip(candidate, self.lb, self.ub), kind

    def _install(self, leaf: Leaf, candidate: np.ndarray, value: float) -> None:
        leaf.replace(
            candidate,
            value,
            species=int(self.rng.integers(0, N_SPECIES)),
            height=float(self.rng.uniform(0.2, 1.0)),
        )
        if value < self.best_fitness:
            self.best_fitness = value
            self.best_position = candidate.copy()

    def _replace_slot(self, leaf: Leaf, tau: float, field_state: float,
                      force_local: bool = False) -> str:
        candidate, kind = self._regrowth_candidate(tau, field_state, force_local)
        value = self.counter(candidate)
        self._install(leaf, candidate, value)
        return kind

    # ------------------------------------------------------------------ #
    # elite refinement
    # ------------------------------------------------------------------ #
    def _elite_step(self, tau: float) -> float:
        initial = float(self.params["elite_step_init"])
        final = float(self.params["elite_step_final"])
        if initial <= 0.0 or final <= 0.0:
            return max(final, initial * (1.0 - tau))
        return initial * (final / initial) ** tau

    def _refine_elites(self, tau: float) -> Tuple[int, int]:
        """Candidate-replacement refinement of the top-K slots.

        Nothing is displaced: a candidate is generated, evaluated, and the
        slot is replaced only if the candidate is strictly better.  Returns
        ``(trials, accepted)``.
        """
        if not self.params["enable_elite"]:
            return 0, 0
        k = min(int(self.params["elite_count"]), len(self.leaves))
        trials_per_leaf = int(self.params["elite_trials"])
        sigma = self._elite_step(tau)
        order = sorted(range(len(self.leaves)), key=lambda i: self.leaves[i].fitness)
        trials = 0
        accepted = 0
        for index in order[:k]:
            leaf = self.leaves[index]
            for _ in range(trials_per_leaf):
                candidate = np.clip(
                    leaf.position + self.rng.normal(0.0, 1.0, self.dim) * sigma * self.span,
                    self.lb,
                    self.ub,
                )
                value = self.counter(candidate)
                trials += 1
                if value < leaf.fitness:
                    leaf.replace(candidate, value)
                    accepted += 1
                    if value < self.best_fitness:
                        self.best_fitness = value
                        self.best_position = candidate.copy()
        return trials, accepted

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _initialise(self) -> None:
        n = int(self.params["population_size"])
        self.leaves = []
        for _ in range(n):
            position = self.lb + self.rng.random(self.dim) * self.span
            leaf = Leaf(
                position=position,
                species=int(self.rng.integers(0, N_SPECIES)),
                height=float(self.rng.uniform(0.2, 1.0)),
            )
            leaf.fitness = self.counter(position)
            self.leaves.append(leaf)
        best = min(self.leaves, key=lambda leaf: leaf.fitness)
        self.best_position = best.position.copy()
        self.best_fitness = best.fitness

    def optimize(self) -> LAOResult:
        start = time.perf_counter()
        self.environment.reset()
        self._vibration_cursor = 0
        self._iteration = 0
        collect = bool(self.params["diagnostics"])
        diag: Dict[str, List[float]] = {
            "nfe": [], "tau": [], "diversity": [], "replacement_rate": [],
            "mean_strength": [], "mean_pressure": [], "crossings": [],
            "elite_trials": [], "elite_accepted": [], "field": [], "gust": [],
            "global_regrowth": [], "vibration_fired": [], "best_error": [],
            "mean_displacement": [], "clip_above_1": [],
        }
        stopped_by_budget = False
        iteration = 0
        max_iter = self.params.get("max_iter")

        try:
            self._initialise()
            while True:
                if self.budget is not None and self.counter.n_evaluations >= self.budget:
                    stopped_by_budget = True
                    break
                if max_iter is not None and iteration >= int(max_iter):
                    break
                iteration += 1
                self._iteration = iteration

                tau = self.progress
                field_state = self.environment.step(tau, self.rng).continuous
                heights = np.asarray([leaf.height for leaf in self.leaves], dtype=float)
                exposure = self.environment.exposure(heights, self.rng)

                positions_before = self._positions() if collect else None
                diversity = self._diversity()
                strength = self._intrinsic_strength()
                pressure = self._abscission_pressure(tau, diversity, exposure)

                crossing = np.flatnonzero(pressure > strength)
                if collect:
                    margin = pressure - strength
                    bins = np.clip(((margin + 1.0) * 10.0).astype(int), 0, 19)
                    np.add.at(self._margin_hist_total, bins, 1.0)
                    replaced_mask = np.zeros(len(self.leaves), dtype=bool)
                    replaced_mask[crossing] = True
                    np.add.at(self._margin_hist_replaced, bins, replaced_mask)
                vibration_fired = self._vibration_due(tau)
                forced: List[int] = []
                if vibration_fired:
                    threshold = float(self.params["vibration_strength_threshold"])
                    probability = float(self.params["vibration_probability"])
                    for index, leaf in enumerate(self.leaves):
                        if (index not in crossing
                                and leaf.strength < threshold
                                and self.rng.random() < probability):
                            forced.append(index)

                replaced = 0
                global_regrowth = 0
                remaining = self.counter.remaining
                if remaining is None:
                    selected_crossing = crossing
                else:
                    selected_crossing = crossing[: int(remaining)]

                for index in selected_crossing:
                    kind = self._replace_slot(self.leaves[int(index)], tau, field_state)
                    replaced += 1
                    global_regrowth += int(kind == "global")

                remaining = self.counter.remaining
                if remaining is None:
                    selected_forced = forced
                else:
                    selected_forced = forced[: int(remaining)]

                for index in selected_forced:
                    self._replace_slot(self.leaves[int(index)], tau, field_state, force_local=True)
                    replaced += 1

                elite_trials, elite_accepted = self._refine_elites(tau)
                self.counter.update_targets()

                if collect:
                    displacement = 0.0
                    if positions_before is not None and self.span_norm > 0:
                        delta = self._positions() - positions_before
                        displacement = float(np.mean(np.linalg.norm(delta, axis=1)) / self.span_norm)
                    diag["nfe"].append(self.counter.n_evaluations)
                    diag["tau"].append(tau)
                    diag["diversity"].append(diversity)
                    diag["replacement_rate"].append(replaced / len(self.leaves))
                    diag["mean_strength"].append(float(np.mean(strength)))
                    diag["mean_pressure"].append(float(np.mean(pressure)))
                    diag["crossings"].append(float(crossing.size))
                    diag["elite_trials"].append(float(elite_trials))
                    diag["elite_accepted"].append(float(elite_accepted))
                    diag["field"].append(float(field_state))
                    diag["gust"].append(float(self.environment.state.gust))
                    diag["global_regrowth"].append(float(global_regrowth))
                    diag["vibration_fired"].append(float(vibration_fired))
                    diag["best_error"].append(self.counter.best_error)
                    diag["mean_displacement"].append(displacement)
                    diag["clip_above_1"].append(self._raw_clip_above)
        except BudgetExhausted:
            stopped_by_budget = True

        trace = self.counter.finalize()
        self.counter.update_targets()
        if self.best_position is None:
            self.best_position = self.counter.best_position
            self.best_fitness = self.counter.best_value
        return LAOResult(
            best_position=np.asarray(self.best_position, dtype=float).copy(),
            best_fitness=float(self.best_fitness),
            best_error=self.counter.best_error,
            n_evaluations=int(self.counter.n_evaluations),
            iterations_run=int(iteration),
            runtime_seconds=time.perf_counter() - start,
            stopped_by_budget=stopped_by_budget,
            params=dict(self.params),
            trace_nfe=trace.grid,
            trace_error=trace.values,
            target_hits=self.counter.target_hits,
            diagnostics={key: np.asarray(value, dtype=float) for key, value in diag.items()} if collect else {},
            margin_hist_total=self._margin_hist_total.copy() if collect else np.zeros(0),
            margin_hist_replaced=self._margin_hist_replaced.copy() if collect else np.zeros(0),
            message="evaluation budget exhausted" if stopped_by_budget else "",
        )

    def _vibration_due(self, tau: float) -> bool:
        schedule = self._vibration_times
        if self._vibration_cursor >= schedule.size:
            return False
        if tau >= schedule[self._vibration_cursor]:
            self._vibration_cursor += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        """Machine-readable description used by the experiment manifest."""
        return {
            "algorithm": self.name,
            "paradigm": "threshold-based selection",
            "components": {name: bool(self.params[f"enable_{name}"]) for name in COMPONENTS},
            "environment": self.environment.describe(),
            "vibration_times": self._vibration_times.tolist(),
            "parameters": dict(self.params),
        }
