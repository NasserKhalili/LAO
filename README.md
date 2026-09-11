# Leaf-Abscission Optimization (LAO)

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2609.10588-b31b1b.svg)](https://arxiv.org/abs/2609.10588)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Leaf-Abscission Optimization (LAO)** is a threshold-based metaheuristic for continuous optimization built around a simple idea: **candidate generation is conditional on a threshold crossing rather than treated as an unconditional step of every iteration.**

The accompanying study develops this idea as an explicit **evaluation-gating** mechanism. For each incumbent, contextual environmental pressure is compared with an incumbent-specific intrinsic strength. A replacement candidate is generated and evaluated only when the pressure exceeds that strength. The result is a selection rule whose replacement activity and objective-evaluation consumption can be measured directly.

> **Nasser Khalili.** *Threshold-Based Selection for Continuous Optimization: A Leaf-Abscission Instantiation.*
>
> Paper: https://arxiv.org/abs/2609.10588
>
> Code: https://github.com/NasserKhalili/LAO

## Why LAO?

Population-based optimizers often couple two different decisions: whether an incumbent should change, and how to construct the candidate that replaces it. LAO separates these decisions.

The core selection event is

$$
\mathrm{AP}_{i,t} > \mathrm{LS}_{i,t},
$$

where $\mathrm{AP}$ denotes contextual environmental pressure and $\mathrm{LS}$ denotes incumbent-specific intrinsic strength. When the threshold is not crossed, no replacement candidate is generated. When it is crossed, the regrowth mechanism constructs and evaluates a replacement under the same budget-aware accounting layer that governs the run.

This formulation provides three useful properties:

- **Explicit selection:** replacement is defined by a measurable pressure–strength crossing.
- **Evaluation-aware search:** objective evaluations are conditional events rather than implicit fixed costs of every iteration.
- **Parsimonious architecture:** the final LAO core retains the threshold-selection mechanism and base regrowth kernel, while optional mechanisms are included only when supported by the ablation evidence.

## What is evaluated?

The accompanying study investigates LAO under controlled objective-evaluation budgets using the official **CEC 2017 single-objective bound-constrained benchmark suite**.

### Primary protocol

| Setting | Protocol |
|---|---|
| Benchmark | CEC 2017: F1 and F3–F30 (F2 excluded) |
| Dimensions | 10, 30, 50 |
| Independent runs | 30 per algorithm/function/dimension |
| Seeds | Shared seeds 0–29 |
| Budget | 300 × D objective evaluations |
| Algorithms | LAO, PSO, GA, DE, GWO, WOA, CMA-ES, SHADE, L-SHADE |

The primary comparison contains **23,490 algorithm/function/dimension runs** under a common evaluation-budget protocol.

### Main result

LAO achieves the **third-best mean Friedman rank at each tested dimension**:

| Dimension | LAO mean Friedman rank |
|---:|---:|
| D = 10 | **4.069** |
| D = 30 | **3.414** |
| D = 50 | **3.069** |

These results support a **competitive, regime-dependent interpretation** of LAO rather than a claim of universal superiority. The relative position of LAO changes with function class and available evaluation budget, with methods such as CMA-ES, SHADE, and L-SHADE becoming stronger in particular regimes.

The study goes beyond final objective values and examines:

- full $2^4$ component ablation of drift, gust, vibration, and elite refinement;
- anytime performance and target-attainment behavior;
- threshold crossings, replacement events, evaluation counts, pressure, strength, diversity, and threshold margins;
- dimension–budget dependence of algorithmic rankings;
- diversity intervention and its effect on replacement activity;
- local sensitivity of key continuous parameters; and
- wall-clock and implementation-level computational cost.

The objective is not only to evaluate **whether LAO is competitive**, but to characterize **when and why its evaluation-gated selection mechanism is effective**.

## Reference implementation

The implementation is organized around a small set of directly testable modules:

```text
src/lao/
├── lao.py           LAO optimizer and threshold-selection logic
├── leaf.py          Leaf state, phenology, and intrinsic strength
├── environment.py   Environmental pressure and forcing model
├── accounting.py    Budget-aware objective evaluation accounting
├── baselines.py     Baseline optimizer implementations
├── cec2017.py       CEC 2017 benchmark interface
└── __init__.py      Public package interface
```

The repository also contains the study configurations, automated implementation/mechanism checks, package-validation utilities, and the documented software environment used for the computational study.

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install LAO in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e .
```

For the extended comparison and testing environment, including CMA-ES and pytest:

```bash
pip install -e ".[full]"
```

## Verification

Run the lightweight implementation checks with:

```bash
python scripts/smoke_test.py
pytest -q
python scripts/validate_package.py
```

## CEC 2017 support data

The CEC 2017 interface expects the official competition support files under:

```text
src/lao/cec2017_data/
```

Place the official `input_data` directory there before running the CEC benchmark protocols. The implementation is designed to use the official reference support files rather than regenerated substitutes.

The protocol files in `configs/` specify the benchmark, ablation, budget, and robustness studies accompanying the paper.

## Reproducibility

This repository provides the reference implementation, study configurations, automated checks, package metadata, and documented environment needed to execute the computational protocols. A complete rerun of the CEC experiments additionally requires the official CEC 2017 support data.

## Citation

If you use LAO or the threshold-selection framework in academic work, please cite the accompanying manuscript:

```text
Khalili, N. (2026). Threshold-Based Selection for Continuous Optimization:
A Leaf-Abscission Instantiation. arXiv:2609.10588.
```

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Research status

The repository accompanies an arXiv preprint and a manuscript submitted for peer review. It is intended as the reference implementation and study-protocol package for the reported empirical analysis.

## License

LAO is released under the MIT License. See [`LICENSE`](LICENSE) for details.
