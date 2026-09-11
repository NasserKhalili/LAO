"""Leaf state and seasonal phenology used by LAO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

#: Logistic slope factor used in the cited phenological model.
LOGISTIC_P3_FACTOR = 2.2

#: Phenological classes and algorithmic strength values used by LAO.
SPECIES_PARAMS: Dict[int, Dict[str, object]] = {
    0: {"name": "early", "peak": 0.30, "spread": 0.10, "strength": 0.60},
    1: {"name": "intermediate", "peak": 0.55, "spread": 0.15, "strength": 0.75},
    2: {"name": "late", "peak": 0.80, "spread": 0.12, "strength": 0.90},
}
N_SPECIES = len(SPECIES_PARAMS)


def cumulative_leaf_fall(tau: float | np.ndarray, peak: float, spread: float,
                         asymptote: float = 1.0) -> float | np.ndarray:
    """Wang et al. (2022) logistic cumulative leaf-fall fraction."""
    slope = LOGISTIC_P3_FACTOR / max(float(spread), 1e-9)
    return asymptote / (1.0 + np.exp(slope * (peak - np.asarray(tau, dtype=float))))


def phenology_landmarks(peak: float, spread: float) -> Dict[str, float]:
    """Derive the start / peak / end / duration landmarks from (P2, P3).

    ``start`` is the 10 % point, ``end`` the 90 % point.  Because the logistic
    is symmetric about ``P2`` on the logit scale, both follow in closed form.
    """
    slope = LOGISTIC_P3_FACTOR / max(float(spread), 1e-9)
    offset = np.log(9.0) / slope
    return {
        "start_10pct": float(peak - float(spread)),
        "peak_50pct": float(peak),
        "end_90pct": float(peak + offset),
        "duration_10_90": float(offset + float(spread)),
    }


def species_seasonal_pressure(tau: float, species: np.ndarray) -> np.ndarray:
    """Vectorised seasonal pressure ``S(tau, s_i)`` for a species vector."""
    peaks = np.array([SPECIES_PARAMS[int(s)]["peak"] for s in species], dtype=float)
    spreads = np.array([SPECIES_PARAMS[int(s)]["spread"] for s in species], dtype=float)
    slope = LOGISTIC_P3_FACTOR / np.maximum(spreads, 1e-9)
    return 1.0 / (1.0 + np.exp(slope * (peaks - float(tau))))


def species_strength(species: np.ndarray) -> np.ndarray:
    """Intrinsic strength scale ``ell(s_i)``."""
    return np.array([SPECIES_PARAMS[int(s)]["strength"] for s in species], dtype=float)


@dataclass
class Leaf:
    """One population slot.

    ``position`` is only ever replaced wholesale by the regrowth operator or
    by an accepted refinement candidate; nothing displaces it in place.  That
    is the stationarity property the threshold paradigm claims, and it is
    enforced by :meth:`replace`.
    """

    position: np.ndarray
    fitness: float = float("inf")
    species: int = 1
    height: float = 1.0
    strength: float = 0.0
    pressure: float = 0.0
    replacements: int = 0

    def replace(self, position: np.ndarray, fitness: float, species: Optional[int] = None,
                height: Optional[float] = None) -> None:
        """Install an accepted candidate in this slot."""
        self.position = position
        self.fitness = float(fitness)
        if species is not None:
            self.species = int(species)
        if height is not None:
            self.height = float(height)
        self.replacements += 1

    @property
    def species_name(self) -> str:
        return str(SPECIES_PARAMS[self.species]["name"])

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"Leaf(species={self.species_name}, f={self.fitness:.4e}, "
                f"LS={self.strength:.3f}, AP={self.pressure:.3f})")


def describe_phenology() -> Dict[str, Dict[str, float]]:
    """Return phenological landmarks for a species class."""
    out: Dict[str, Dict[str, float]] = {}
    for key, params in SPECIES_PARAMS.items():
        landmarks = phenology_landmarks(float(params["peak"]), float(params["spread"]))
        landmarks["strength"] = float(params["strength"])
        out[str(params["name"])] = landmarks
    return out
