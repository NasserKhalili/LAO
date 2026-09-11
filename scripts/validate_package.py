from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    modules = [
        "lao",
        "lao.accounting",
        "lao.baselines",
        "lao.cec2017",
        "lao.environment",
        "lao.lao",
        "lao.leaf",
    ]
    for name in modules:
        importlib.import_module(name)
    print("imports: PASS")

    from lao import DEFAULT_PARAMS, LAO
    from lao.accounting import CountingObjective
    import numpy as np

    dim = 10
    budget = 300
    bounds = (np.full(dim, -100.0), np.full(dim, 100.0))
    objective = CountingObjective(lambda x: float(np.sum(np.asarray(x) ** 2)), budget=budget)
    result = LAO(objective, dim, bounds, params={"max_evaluations": budget, "seed": 0}).optimize()
    assert result.n_evaluations == budget
    assert result.stopped_by_budget
    assert all(DEFAULT_PARAMS[f"enable_{c}"] is False for c in ("drift", "gust", "vibration", "elite"))
    assert result.best_position.shape == (dim,)
    assert np.isfinite(result.best_fitness)
    print("LAO smoke: PASS")


if __name__ == "__main__":
    main()
