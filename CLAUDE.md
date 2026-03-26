# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`idinn` (Inventory Dynamics-Informed Neural Networks) is a Python package for optimizing inventory systems using neural network-based controllers and traditional baselines (base-stock policies, dynamic programming). It targets single-sourcing and dual-sourcing supply chain problems using PyTorch as the computational backbone.

## Commands

**Setup (uses `uv` package manager):**
```bash
uv sync --group dev --python 3.12
```

**Run tests:**
```bash
uv run pytest tests/
# Single test file:
uv run pytest tests/dual_sourcing_controller/test_dual_neural.py
# With coverage:
python -m coverage run -m pytest && python -m coverage report
```

**Lint and type check:**
```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

**Docker alternatives:**
```bash
docker compose up tests    # Run tests in container
docker compose up app      # Streamlit app at port 8501
docker compose up juplab   # Jupyter Lab at port 8888
```

**Build docs:**
```bash
cd docs && make html
```

## Architecture

### Layer structure

```
Sourcing Models  →  Controllers  →  Training/Prediction
(demand, costs,     (neural NN,      (fit, predict,
 lead times)         base-stock,      get_total_cost)
                     DP solvers)
```

### Core modules (`src/idinn/`)

- **`sourcing_model.py`** — `BaseSourcingModel`, `SingleSourcingModel`, `DualSourcingModel`: define inventory dynamics (demand sampling, lead time logic, cost accumulation). All state is PyTorch tensors.
- **`demand.py`** — `BaseDemand` (ABC), `UniformDemand`, `CustomDemand`: pluggable demand generators.
- **`single_controller/`** — `SingleSourcingNeuralController` (NN-based), `BaseStockController` (heuristic baseline). Abstract base in `base.py`.
- **`dual_controller/`** — `DualSourcingNeuralController` (NN-based), `CappedDualIndexController` (heuristic), `DynamicProgrammingController` (optimal). Abstract base in `base.py`.

### Training workflow

1. Create a sourcing model with demand, costs, and lead times.
2. Instantiate a controller (neural or baseline).
3. Call `controller.fit(sourcing_model, ...)` — trains with PyTorch SGD/Adam; logs to TensorBoard.
4. Call `controller.predict(current_inventory)` to generate order quantities.
5. Evaluate with `controller.get_total_cost(sourcing_model, ...)`.

### Key design details

- All inventory state, orders, and demands flow as PyTorch tensors — enables batch processing and GPU acceleration.
- Neural controllers have configurable hidden layers and activation functions (e.g., `torch.nn.CELU()`).
- TensorBoard logging is built into the training loop.
- Numba is used for JIT compilation in performance-critical DP solvers.

### Streamlit app (`app/`)

Interactive UI for training and evaluating controllers. `app/utils.py` provides `tflog2pandas` to convert TensorBoard logs for visualization with Plotly.

## Code Style

- NumPy-style docstrings required.
- PEP 8 enforced via `ruff`; type annotations enforced via `mypy`.
- Tests are required for all new functionality.
