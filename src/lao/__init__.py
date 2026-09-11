"""Leaf-Abscission Optimization for continuous black-box optimization."""

from .accounting import (
    DEFAULT_TARGETS,
    BudgetExhausted,
    CountingObjective,
    EvaluationTrace,
    geometric_trace_grid,
)
from .baselines import (
    ALGORITHMS as BASELINE_ALGORITHMS,
    CMAES,
    DE,
    GA,
    GWO,
    LSHADE,
    PSO,
    SHADE,
    WOA,
    OptimizerResult,
)
from .cec2017 import (
    EXCLUDED_FUNCTIONS,
    EXPERIMENTAL_FUNCTIONS,
    IS_OFFICIAL_CEC2017,
    CEC2017Function,
    CEC2017Suite,
)
from .environment import EnvironmentField, FORCING_PROCESSES, PROFILES
from .lao import COMPONENTS, DEFAULT_PARAMS, LAO, LAOResult
from .leaf import Leaf, SPECIES_PARAMS, describe_phenology

__version__ = "1.1.0"

OPTIMIZERS = {"LAO": LAO, **BASELINE_ALGORITHMS}

__all__ = [
    "BASELINE_ALGORITHMS", "BudgetExhausted", "CEC2017Function",
    "CEC2017Suite", "CMAES", "COMPONENTS", "CountingObjective", "DE",
    "DEFAULT_PARAMS", "DEFAULT_TARGETS", "EnvironmentField", "EXCLUDED_FUNCTIONS",
    "EXPERIMENTAL_FUNCTIONS", "EvaluationTrace", "FORCING_PROCESSES", "GA",
    "GWO", "IS_OFFICIAL_CEC2017", "LAO", "LAOResult", "LSHADE", "Leaf",
    "OPTIMIZERS", "OptimizerResult", "PROFILES", "PSO", "SHADE",
    "SPECIES_PARAMS", "WOA", "__version__", "describe_phenology",
    "geometric_trace_grid",
]
