from __future__ import annotations

import numpy as np

from lao import CountingObjective, LAO


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def main() -> None:
    dimension = 10
    budget = 300
    bounds = (np.full(dimension, -100.0), np.full(dimension, 100.0))
    counter = CountingObjective(sphere, budget=budget, optimum=0.0)
    optimizer = LAO(
        counter,
        dimension,
        bounds,
        params={"max_evaluations": budget, "seed": 0},
    )
    result = optimizer.optimize()
    print(f"best_fitness={result.best_fitness:.6e}")
    print(f"n_evaluations={result.n_evaluations}/{budget}")
    print(f"iterations={result.iterations_run}")
    if result.n_evaluations != budget:
        raise SystemExit("Smoke test did not consume the requested evaluation budget.")


if __name__ == "__main__":
    main()
