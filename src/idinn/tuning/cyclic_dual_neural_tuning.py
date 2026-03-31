"""
Hyperparameter optimisation for CyclicDualSourcingNeuralController using Optuna.

Usage
-----
# Install tuning dependencies first:
#   uv sync --group tuning

# Run:
#   uv run python src/idinn/tuning/cyclic_dual_neural_tuning.py

# Visualise results (requires optuna-dashboard):
#   optuna-dashboard sqlite:///tuning.db
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Logging — writes to both stdout and a log file next to this script
# ---------------------------------------------------------------------------
_LOG_FILE = Path(__file__).parent / "cyclic_dual_neural_tuning.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — edit these to change the run
# ---------------------------------------------------------------------------

# Problem definition
PROBLEM = dict(
    regular_lead_time=3,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=0,
    demand_low=0,
    demand_high=4,
    train_batch_size=32,
)

CYCLE_LENGTH = 2          # 1, 2, or 3
N_TRIALS = 60             # number of Optuna trials
SEARCH_EPOCHS = 400       # training epochs per trial (cheap pass)
SEARCH_PERIODS = 150      # sourcing periods per epoch during search
FINAL_EPOCHS = 2000       # epochs for full retrain of best config
FINAL_PERIODS = 300       # sourcing periods per epoch for final retrain
STORAGE = "sqlite:///tuning.db"   # set to None for in-memory (not resumable)
OUT_DIR = Path(__file__).parent / "results"
RUN_DP_BASELINE = True    # compare against DP optimal after tuning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sourcing_model(batch_size: int):
    """Return a freshly initialised DualSourcingModel from PROBLEM config."""
    from idinn.demand import UniformDemand
    from idinn.sourcing_model import DualSourcingModel

    return DualSourcingModel(
        regular_lead_time=PROBLEM["regular_lead_time"],
        expedited_lead_time=PROBLEM["expedited_lead_time"],
        regular_order_cost=PROBLEM["regular_order_cost"],
        expedited_order_cost=PROBLEM["expedited_order_cost"],
        holding_cost=PROBLEM["holding_cost"],
        shortage_cost=PROBLEM["shortage_cost"],
        init_inventory=PROBLEM["init_inventory"],
        demand_generator=UniformDemand(
            low=PROBLEM["demand_low"], high=PROBLEM["demand_high"]
        ),
        batch_size=batch_size,
    )


def build_hidden_layers(trial) -> list:
    """
    Sample a hidden layer configuration.

    Produces a geometrically shrinking list of widths, e.g. [128, 64, 32].
    """
    first = trial.suggest_categorical("first_layer_size", [32, 64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    shrink = trial.suggest_categorical("shrink_factor", [1, 2])  # 1 = constant width
    layers = []
    size = first
    for _ in range(n_layers):
        layers.append(max(size, 4))
        size = max(size // shrink, 4)
    return layers


def build_activation(trial) -> torch.nn.Module:
    name = trial.suggest_categorical("activation", ["relu", "celu", "elu", "tanh"])
    return {
        "relu": torch.nn.ReLU(),
        "celu": torch.nn.CELU(alpha=1.0),
        "elu": torch.nn.ELU(),
        "tanh": torch.nn.Tanh(),
    }[name]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(cycle_length: int, search_epochs: int, search_periods: int):
    """Return a closure that Optuna calls as objective(trial)."""

    def objective(trial):
        import optuna
        from idinn.cyclic_dual_controller.cyclic_dual_neural import (
            CyclicDualSourcingNeuralController,
        )

        hidden_layers = build_hidden_layers(trial)
        activation = build_activation(trial)
        compressed = trial.suggest_categorical("compressed", [False, True])
        parameters_lr = trial.suggest_float("parameters_lr", 1e-4, 1e-2, log=True)
        init_inventory_lr = trial.suggest_float("init_inventory_lr", 5e-3, 5e-1, log=True)
        init_inventory_freq = trial.suggest_int("init_inventory_freq", 2, 8)

        sourcing_model = make_sourcing_model(batch_size=PROBLEM["train_batch_size"])
        controller = CyclicDualSourcingNeuralController(
            cycle_length=cycle_length,
            hidden_layers=hidden_layers,
            activation=activation,
            compressed=compressed,
        )
        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=search_periods,
            epochs=search_epochs,
            parameters_lr=parameters_lr,
            init_inventory_lr=init_inventory_lr,
            init_inventory_freq=init_inventory_freq,
            seed=42,
        )

        eval_model = make_sourcing_model(batch_size=1)
        avg_cost = controller.get_average_cost(
            eval_model, sourcing_periods=500, seed=0
        ).item()

        trial.report(avg_cost, step=search_epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return avg_cost

    return objective


# ---------------------------------------------------------------------------
# Final retraining with the best config
# ---------------------------------------------------------------------------

def retrain_best(best_params: dict, cycle_length: int, out_dir: Path) -> float:
    """Retrain from scratch with best hyperparameters at full budget."""
    from idinn.cyclic_dual_controller.cyclic_dual_neural import (
        CyclicDualSourcingNeuralController,
    )

    first = best_params["first_layer_size"]
    n_layers = best_params["n_layers"]
    shrink = best_params["shrink_factor"]
    size = first
    hidden_layers = []
    for _ in range(n_layers):
        hidden_layers.append(max(size, 4))
        size = max(size // shrink, 4)

    activation = {
        "relu": torch.nn.ReLU(),
        "celu": torch.nn.CELU(alpha=1.0),
        "elu": torch.nn.ELU(),
        "tanh": torch.nn.Tanh(),
    }[best_params["activation"]]

    sourcing_model = make_sourcing_model(batch_size=PROBLEM["train_batch_size"])
    controller = CyclicDualSourcingNeuralController(
        cycle_length=cycle_length,
        hidden_layers=hidden_layers,
        activation=activation,
        compressed=best_params["compressed"],
    )
    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=FINAL_PERIODS,
        epochs=FINAL_EPOCHS,
        parameters_lr=best_params["parameters_lr"],
        init_inventory_lr=best_params["init_inventory_lr"],
        init_inventory_freq=best_params["init_inventory_freq"],
        seed=42,
    )

    eval_model = make_sourcing_model(batch_size=1)
    avg_cost = controller.get_average_cost(
        eval_model, sourcing_periods=1000, seed=0
    ).item()
    logger.info(f"Final retrained avg cost/period: {avg_cost:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"best_model_n{cycle_length}.pt"
    controller.save(str(model_path))
    logger.info(f"Saved best model to {model_path}")

    return avg_cost


# ---------------------------------------------------------------------------
# DP baseline
# ---------------------------------------------------------------------------

def run_dp_baseline(cycle_length: int) -> Optional[float]:
    """Run the DP controller as a cost baseline."""
    try:
        from idinn.cyclic_dual_controller.dynamic_programming import (
            DynamicProgrammingController,
        )

        sourcing_model = make_sourcing_model(batch_size=1)
        dp = DynamicProgrammingController(cycle_length=cycle_length)
        dp.fit(sourcing_model=sourcing_model, max_iterations=5000, tolerance=10e-7)
        eval_model = make_sourcing_model(batch_size=1)
        cost = dp.get_average_cost(eval_model, sourcing_periods=1000, seed=0).item()
        logger.info(f"DP baseline avg cost/period (N={cycle_length}): {cost:.4f}")
        return cost
    except Exception as e:
        logger.warning(f"DP baseline failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    try:
        import optuna
    except ImportError:
        logger.error("optuna is not installed. Run: uv sync --group tuning")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"cyclic_neural_n{CYCLE_LENGTH}_{timestamp}"
    out_dir = OUT_DIR / study_name

    logger.info("=" * 60)
    logger.info(f"Study: {study_name}")
    logger.info(f"Cycle length N={CYCLE_LENGTH}")
    logger.info(f"Trials: {N_TRIALS}  |  Search epochs: {SEARCH_EPOCHS}")
    logger.info(f"Problem: {PROBLEM}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: hyperparameter search
    # ------------------------------------------------------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    sampler = optuna.samplers.TPESampler(seed=0)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=STORAGE,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        make_objective(CYCLE_LENGTH, SEARCH_EPOCHS, SEARCH_PERIODS),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial #{best.number}  value={best.value:.4f}")
    logger.info(f"Best params: {best.params}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 2: full retrain with best config
    # ------------------------------------------------------------------
    logger.info("Retraining best configuration at full budget...")
    final_cost = retrain_best(best.params, CYCLE_LENGTH, out_dir)

    # ------------------------------------------------------------------
    # Optional DP baseline
    # ------------------------------------------------------------------
    dp_cost = None
    if RUN_DP_BASELINE:
        logger.info("Running DP baseline for comparison...")
        dp_cost = run_dp_baseline(CYCLE_LENGTH)

    # ------------------------------------------------------------------
    # Save results summary
    # ------------------------------------------------------------------
    results = {
        "study_name": study_name,
        "cycle_length": CYCLE_LENGTH,
        "problem": PROBLEM,
        "best_trial": best.number,
        "best_search_cost": best.value,
        "best_final_cost": final_cost,
        "dp_baseline_cost": dp_cost,
        "gap_vs_dp_pct": (
            round((final_cost - dp_cost) / dp_cost * 100, 2)
            if dp_cost is not None else None
        ),
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    if dp_cost is not None:
        logger.info(f"Neural vs DP gap: {results['gap_vs_dp_pct']:+.2f}%")


if __name__ == "__main__":
    main()
