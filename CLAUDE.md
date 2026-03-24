# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`idinn` is a Python package for inventory-dynamics control with neural networks. It implements neural network controllers and classical/DP baselines for single-sourcing and dual-sourcing inventory optimization problems.

## Development Commands

### Setup
```bash
uv sync --group dev --python 3.12
```

### Testing
```bash
uv run pytest tests/                          # run all tests
uv run pytest tests/phase_controller/         # run a specific test directory
python -m coverage run -m pytest && python -m coverage report  # with coverage
```

### Linting & Type Checking
```bash
uv run ruff check .       # lint
uv run ruff format .      # format
uv run mypy src/          # type check
```

### Running the Streamlit App
```bash
streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
```

### Docker (with GPU support)
```bash
docker compose up tests   # run tests in container
docker compose up app     # run Streamlit app
docker compose up juplab  # run JupyterLab
```

## Architecture

### Core Abstractions

**Demand Models** (`src/idinn/demand.py`): Abstract `BaseDemand` with `UniformDemand` and `CustomDemand` implementations. These generate stochastic demand samples used during simulation/training.

**Sourcing Models** (`src/idinn/sourcing_model.py`): `SingleSourcingModel` and `DualSourcingModel` simulate inventory dynamics. `DualSourcingModel` has two suppliers — regular (cheaper, longer lead time) and expedited (expensive, shorter lead time).

**Controllers** — decision policies that determine how much to order each period:
- `src/idinn/single_controller/` — single-sourcing policies:
  - `BaseStockController` — classical base-stock policy
  - `SingleSourcingNeuralController` — trainable PyTorch NN policy
- `src/idinn/dual_controller/` — dual-sourcing policies:
  - `DualSourcingNeuralController` — NN policy
  - `DynamicProgrammingController` — optimal DP baseline
  - `CappedDualIndexController` — capped dual index heuristic
- `src/idinn/phase_controller/` — multi-period cyclic planning:
  - `neural/` — multi-head and multi-period NN controllers with Optuna HPO
  - `dp/` — DP-based phase controllers

### Training Pattern

Neural controllers are trained end-to-end via gradient descent on simulated costs. The training loop:
1. Sample demand from a `Demand` model
2. Simulate inventory dynamics using a `SourcingModel`
3. Compute total cost (holding + backlog + ordering)
4. Backpropagate through the simulation to update controller weights

TensorBoard is integrated for training visualization.

### Scripts

`scripts/` contains standalone research scripts for hyperparameter optimization (HPO) and fine-tuning. These are not part of the installed package.

### Key Design Conventions

- NumPy-style docstrings throughout
- Abstract base classes define the controller/model interfaces
- PyTorch tensors used for batched simulation (multiple demand trajectories in parallel)
- `numba` used for performance-critical DP computations
