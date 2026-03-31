# idinn: Inventory-Dynamics Control with Neural Networks

[![PyPI Latest Release](https://img.shields.io/pypi/v/idinn.svg)](https://pypi.org/project/idinn/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BAMiveGXmErIp10MK3V_SUJlDAXHAyaI)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idinn-demo.streamlit.app)
[![status](https://joss.theoj.org/papers/224380be40f3be0b741a4ec711eac83b/status.svg)](https://joss.theoj.org/papers/224380be40f3be0b741a4ec711eac83b)

[<img src="https://gitlab.com/ComputationalScience/idinn/-/raw/main/docs/_static/youtube.png" align="center" width="60%" size="auto" alt="youtube">](https://www.youtube.com/watch?v=hUBfTWV6tWQ)

`idinn` implements **i**nventory **d**ynamics–**i**nformed **n**eural **n**etwork and other related controllers for solving single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into customizable objects using PyTorch as backend to enable users to find the optimal controllers for user-specified inventory systems.

## Demo

For a quick demo, run our [Streamlit app](https://idinn-demo.streamlit.app/). The app allows you to interactively train and evaluate neural controllers for user-specified dual-sourcing systems. Alternatively, use our notebook in [Colab](https://colab.research.google.com/drive/1BAMiveGXmErIp10MK3V_SUJlDAXHAyaI).

---

## Table of Contents

- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Sourcing Models](#sourcing-models)
- [Demand Generators](#demand-generators)
- [Controllers](#controllers)
  - [Single-Sourcing Controllers](#single-sourcing-controllers)
  - [Dual-Sourcing Controllers](#dual-sourcing-controllers)
  - [Cyclic Dual-Sourcing Controllers](#cyclic-dual-sourcing-controllers)
- [Example Usage](#example-usage)
- [Cyclic Neural Controller & Seed Tuning CLI](#cyclic-neural-controller--seed-tuning-cli)
- [Checkpoint Management](#checkpoint-management)
- [Streamlit App](#streamlit-app)
- [Development Setup](#development-setup)
- [Documentation](#documentation)
- [Papers](#papers-using-idinn)
- [Contributors](#contributors)

---

## Installation

Install from PyPI:

```bash
pip install idinn
```

To inspect or edit source locally:

```bash
git clone https://gitlab.com/ComputationalScience/idinn.git
cd idinn
pip install -e .
```

For development (includes test, lint, and doc dependencies):

```bash
uv sync --group dev --python 3.12
```

---

## Architecture Overview

```
Sourcing Models  →  Controllers  →  Training / Prediction
(demand, costs,     (neural NN,      (fit, predict,
 lead times)         base-stock,      get_total_cost,
                     DP solvers,      simulate, plot)
                     cyclic NN)
```

All inventory state, orders, and demands flow as PyTorch tensors, enabling batch processing and GPU acceleration. TensorBoard logging is built into every neural training loop.

---

## Sourcing Models

Sourcing models define inventory dynamics: how demand is realized, how orders arrive, and how costs accumulate. They maintain all state as PyTorch tensors.

### `SingleSourcingModel`

One supplier with a fixed lead time.

```python
from idinn.sourcing_model import SingleSourcingModel
from idinn.demand import UniformDemand

model = SingleSourcingModel(
    lead_time=2,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=10,
    demand_generator=UniformDemand(low=0, high=4),
    batch_size=32,
)
```

| Parameter | Description |
|---|---|
| `lead_time` | Periods before a placed order arrives |
| `holding_cost` | Per-unit cost for positive inventory |
| `shortage_cost` | Per-unit cost for negative inventory (backorders) |
| `init_inventory` | Starting inventory level (optimized jointly during training) |
| `demand_generator` | A `BaseDemand` instance |
| `batch_size` | Number of parallel simulation trajectories |

### `DualSourcingModel`

Two suppliers: a slow (regular) supplier and a fast (expedited) supplier. The expedited lead time is typically 0 (immediate arrival).

```python
from idinn.sourcing_model import DualSourcingModel

model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4),
    batch_size=1,
)
```

| Parameter | Description |
|---|---|
| `regular_lead_time` | Lead time for the regular (slow) supplier |
| `expedited_lead_time` | Lead time for the expedited (fast) supplier |
| `regular_order_cost` | Per-unit cost for regular orders |
| `expedited_order_cost` | Per-unit cost for expedited orders |

---

## Demand Generators

Demand generators are pluggable objects that implement `BaseDemand`.

### `UniformDemand`

Samples integer demand uniformly at random.

```python
from idinn.demand import UniformDemand

demand = UniformDemand(low=0, high=4)   # demand ∈ {0, 1, 2, 3, 4}
```

### `CustomDemand`

Samples demand from a user-specified discrete distribution.

```python
from idinn.demand import CustomDemand

demand = CustomDemand(demand_prob={0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2})
```

The probabilities must sum to 1.0 and all keys must be non-negative integers.

---

## Controllers

### Single-Sourcing Controllers

#### `SingleSourcingNeuralController`

A feedforward neural network that maps `(current_inventory, past_orders)` to an integer order quantity.

```python
import torch
from idinn.single_controller import SingleSourcingNeuralController

controller = SingleSourcingNeuralController(
    hidden_layers=[2],
    activation=torch.nn.CELU(alpha=1),
)
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    epochs=5000,
    validation_sourcing_periods=1000,
    seed=1,
)
controller.predict(current_inventory=10, past_orders=[2, 3])
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
```

The network input is `[current_inventory, past_orders[-lead_time:]]`. Output is a non-negative integer (ReLU + straight-through floor estimator).

#### `BaseStockController`

Classical order-up-to heuristic. Orders up to the optimal base-stock level `z*`, computed analytically from the demand distribution.

```python
from idinn.single_controller import BaseStockController

controller = BaseStockController()
controller.fit(sourcing_model=sourcing_model, num_samples=100000, seed=0)
controller.predict(current_inventory=10, past_orders=[2, 3])
```

The optimal level is `z* = F^{-1}(b / (b + h))` where `F` is the CDF of cumulative (L+1)-period demand.

---

### Dual-Sourcing Controllers

#### `DualSourcingNeuralController`

Neural network controller for dual sourcing. Outputs `(regular_q, expedited_q)` each period.

```python
from idinn.dual_controller import DualSourcingNeuralController

controller = DualSourcingNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4],
    activation=torch.nn.ReLU(),
)
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=100,
    epochs=5000,
    validation_sourcing_periods=1000,
    seed=1,
)
regular_q, expedited_q = controller.predict(
    current_inventory=6,
    past_regular_orders=[2, 3],
    past_expedited_orders=[1],
)
```

#### `DynamicProgrammingController` (dual sourcing)

Computes the optimal policy via value iteration. Requires `UniformDemand` and `expedited_lead_time=0`.

```python
from idinn.dual_controller import DynamicProgrammingController

controller = DynamicProgrammingController()
controller.fit(sourcing_model=sourcing_model, max_iterations=1000000, tolerance=1e-7)
regular_q, expedited_q = controller.predict(
    current_inventory=6,
    past_regular_orders=[2, 3],
    past_expedited_orders=[1],
)
```

Uses Numba JIT compilation for the inner Bellman update loop.

#### `CappedDualIndexController`

Grid-search heuristic based on the Capped Dual Index policy (Sun & Van Mieghem 2019). Parameterized by thresholds `(s_e, s_r)` and a regular order cap `q_r`.

```python
from idinn.dual_controller import CappedDualIndexController

controller = CappedDualIndexController()
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=100,
    s_e_range=range(0, 20),
    s_r_range=range(0, 20),
    q_r_range=range(1, 10),
    seed=0,
)
```

---

### Cyclic Dual-Sourcing Controllers

Cyclic controllers operate on multi-period cycles of length `n_cycles`. Within each cycle, a single forward pass determines all order quantities for the entire cycle. This reduces the decision frequency and allows the network to plan ahead across multiple periods.

#### `CyclicDualNeuralController`

The network takes the current inventory state and outputs `n_cycles + 1` order quantities:
- `(q_r_0, q_e_0, q_e_1, ..., q_e_{n_cycles-1})`
- Regular order `q_r_0` is placed only in period 0 of the cycle.
- Expedited orders `q_e_k` are placed in each of the `n_cycles` periods.

**Example with `n_cycles=2`** (outputs 3 values: `q_r_0, q_e_0, q_e_1`):

```python
import torch
from idinn.cyclic_dual_controller import CyclicDualNeuralController
from idinn.sourcing_model import DualSourcingModel
from idinn.demand import UniformDemand

sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4),
    batch_size=1,
)

controller = CyclicDualNeuralController(
    hidden_layers=[64, 32, 16, 8],
    activation=torch.nn.CELU(alpha=1.0),
    n_cycles=2,
)
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=100,
    epochs=8500,
    validation_sourcing_periods=1000,
    parameters_lr=3e-4,
    seed=42,
    checkpoint_path="models/trained/best_model.pt",
)
```

**Key training details:**
- Optimizer: Adam with cosine annealing LR schedule
- Gradient clipping: `max_norm=1.0`
- Saves the best checkpoint (by validation cost) to `checkpoint_path` during training

#### `DynamicProgrammingController` (cyclic)

Optimal cyclic policy via value iteration. Supports cycle lengths 1, 2, and 3.

```python
from idinn.cyclic_dual_controller import DynamicProgrammingController

controller = DynamicProgrammingController(cycle_length=2)
controller.fit(sourcing_model=sourcing_model)
```

---

## Example Usage

### Single Sourcing

```python
import torch
from idinn.sourcing_model import SingleSourcingModel
from idinn.single_controller import SingleSourcingNeuralController
from idinn.demand import UniformDemand

sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=10,
    demand_generator=UniformDemand(low=1, high=4),
)
controller = SingleSourcingNeuralController(
    hidden_layers=[2],
    activation=torch.nn.CELU(alpha=1)
)
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    seed=1,
)
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
controller.predict(current_inventory=10)
```

### Dual Sourcing

```python
from idinn.sourcing_model import DualSourcingModel
from idinn.dual_controller import DualSourcingNeuralController
from idinn.demand import UniformDemand

sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4),
    batch_size=1,
)
controller = DualSourcingNeuralController(hidden_layers=[128, 64, 32, 16, 8, 4])
controller.fit(sourcing_model=sourcing_model, sourcing_periods=100, epochs=5000, seed=1)
regular_q, expedited_q = controller.predict(
    current_inventory=6,
    past_regular_orders=[2, 3],
    past_expedited_orders=[1],
)
```

---

## Cyclic Neural Controller & Seed Tuning CLI

`src/idinn/tuning/seed_tuning.py` provides a command-line interface for training and evaluating `CyclicDualNeuralController` across multiple random seeds. It trains the hardcoded dual-sourcing problem:

| Parameter | Value |
|---|---|
| `regular_lead_time` | 2 |
| `expedited_lead_time` | 0 |
| `regular_order_cost` | 0 |
| `expedited_order_cost` | 20 |
| `holding_cost` | 5 |
| `shortage_cost` | 495 |
| `init_inventory` | 6 |
| `demand_generator` | `UniformDemand(0, 4)` |

**Run from the repository root** (not from inside `src/`):

```bash
# Train a single model with seed=42
python src/idinn/tuning/seed_tuning.py --train

# Train 10 models (seeds 0–9), evaluate each, save the best
python src/idinn/tuning/seed_tuning.py --seed_train

# Load the best checkpoint and run inference
python src/idinn/tuning/seed_tuning.py --infer
```

### `--train`

Trains a single `CyclicDualNeuralController` with `seed=42`, `hidden_layers=[64, 32, 16, 8]`, `n_cycles=2`, `epochs=8500`, `sourcing_periods=100`.

Saves the checkpoint to:
```
models/trained/best_model.pt
```

Raises `RuntimeError` if the file already exists (to prevent accidental overwriting).

### `--seed_train`

Trains 10 models with seeds 0–9. For each seed:
1. Trains with the same hyperparameters as `--train`, saving to `models/trained/seeded/model{i}.pt`.
2. Evaluates over 50 random seeds × 1000 periods.
3. Copies the checkpoint with the lowest mean cost to `models/trained/best_model.pt`.

Skips training for any seed whose checkpoint already exists (resume-safe).

Checkpoint layout after `--seed_train`:
```
models/
└── trained/
    ├── best_model.pt          ← copy of the best seed's checkpoint
    └── seeded/
        ├── model1.pt          ← seed 0
        ├── model2.pt          ← seed 1
        ├── ...
        └── model10.pt         ← seed 9
```

### `--infer`

Loads `models/trained/best_model.pt` and evaluates over 100 random seeds × 1000 periods.

**The checkpoint must exist at `models/trained/best_model.pt` before running `--infer`.** Create it with either `--train` or `--seed_train` first:

```bash
# Option A: quick single-seed train
python src/idinn/tuning/seed_tuning.py --train

# Option B: multi-seed train to find the best model
python src/idinn/tuning/seed_tuning.py --seed_train

# Then run inference
python src/idinn/tuning/seed_tuning.py --infer
```

Training logs are written to `src/idinn/tuning/tune_cyclic_neural.log`.

---

## Checkpoint Management

`CyclicDualNeuralController` has built-in checkpoint save/load methods.

### Saving

Checkpoints are saved automatically during `fit()` (whenever validation cost improves) if `checkpoint_path` is provided. You can also save manually:

```python
controller.save_checkpoint("path/to/checkpoint.pt")
```

The checkpoint stores: `model_state_dict`, `hidden_layers`, `n_cycles`, and `init_inventory`.

### Loading

```python
sourcing_model = get_sourcing_model()  # reconstruct the sourcing model
controller = CyclicDualNeuralController.load_checkpoint(
    path="models/trained/best_model.pt",
    sourcing_model=sourcing_model,
)
```

`load_checkpoint` restores `hidden_layers`, `n_cycles`, and `init_inventory` from the checkpoint. The `sourcing_model` must match the one used during training (same lead times, costs, and demand).

> **Note:** Checkpoints saved before `n_cycles` was added to the checkpoint format will default to `n_cycles=2` when loaded.

---

## Streamlit App

An interactive web UI for training and evaluating controllers:

```bash
# Local
streamlit run app/app.py

# Docker
docker compose up app   # available at http://localhost:8501
```

The app supports real-time training logs, TensorBoard curve visualization (via `app/utils.py:tflog2pandas`), model architecture display, and simulation plots.

---

## Development Setup

```bash
# Install all dev dependencies
uv sync --group dev --python 3.12

# Run tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/dual_sourcing_controller/test_dual_neural.py

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/

# Build docs
cd docs && make html

# Docker alternatives
docker compose up tests    # run tests
docker compose up juplab   # Jupyter Lab at port 8888
```

---

## Documentation

See the official [documentation](https://inventory-optimization.readthedocs.io/en/latest/) for full API reference and tutorials.

---

## Papers using `idinn`

* Böttcher, Lucas, Thomas Asikis, and Ioannis Fragkos. "Control of dual-sourcing inventory systems using recurrent neural networks." [INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2022.0136) 35.6 (2023): 1308-1328.
* Li, Jiawei, Thomas Asikis, Ioannis Fragkos, and Böttcher, Lucas. "idinn: A Python package for inventory-dynamics control with neural networks." [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.08508#) 10.112 (2025): 8508.

```bibtex
@article{bottcher2023control,
  title={Control of dual-sourcing inventory systems using recurrent neural networks},
  author={B{\"o}ttcher, Lucas and Asikis, Thomas and Fragkos, Ioannis},
  journal={INFORMS Journal on Computing},
  volume={35},
  number={6},
  pages={1308--1328},
  year={2023}
}
```

```bibtex
@article{li2025, 
  title = {idinn: A {P}ython package for inventory-dynamics control with neural networks}, 
  author = {Li, Jiawei and Asikis, Thomas and Fragkos, Ioannis and B{\"o}ttcher, Lucas}, 
  journal = {Journal of Open Source Software},
  volume = {10}, 
  number = {112}, 
  pages = {8508}, 
  year = {2025}
}
```

## Contributors

* [Jiawei Li](https://github.com/iewaij)
* [Thomas Asikis](https://gitlab.com/asikist)
* [Ioannis Fragkos](https://gitlab.com/ioannis.fragkos1)
* [Lucas Böttcher](https://gitlab.com/lucasboettcher)
