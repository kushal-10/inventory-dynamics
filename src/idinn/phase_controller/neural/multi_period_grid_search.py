"""
Hyperparameter Optimisation for MultiPeriodNeuralController using Optuna.

Search space:
  - Architecture : predefined non-increasing layer configs
  - Optimizer    : learning rate

Optimisation target : training cost (lower is better).
Sampler             : TPE
Pruner              : NopPruner (all trials run to completion)
Trials              : 25
"""

import logging

import torch
import optuna
from optuna.samplers import TPESampler

from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="src/idinn/phase_controller/neural/multi_period_optuna.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Sourcing model  (expedited_lead_time=0, regular_order_cost=0 to match DP)
# ─────────────────────────────────────────────────────────────────────────────

sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    init_inventory=6,
    demand_generator=UniformDemand(0, 4),
    batch_size=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Search-space helpers
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATION = torch.nn.CELU(alpha=1.0)  # fixed, not tuned

# Predefined architectures — all non-increasing, covering shallow/deep and wide/narrow
ARCHITECTURES = {
    # Wide and shallow
    "128_64":          [128, 64],
    "128_64_32":       [128, 64, 32],
    "128_64_32_16":    [128, 64, 32, 16],
    "128_64_32_16_8":  [128, 64, 32, 16, 8],
    # Medium width
    "64_32":           [64, 32],
    "64_32_16":        [64, 32, 16],
    "64_32_16_8":      [64, 32, 16, 8],
    "64_32_16_8_4":    [64, 32, 16, 8, 4],
    # Narrower
    "32_16":           [32, 16],
    "32_16_8":         [32, 16, 8],
    "32_16_8_4":       [32, 16, 8, 4],
    # Deep and narrow
    "16_8":            [16, 8],
    "16_8_4":          [16, 8, 4],
    # Wide with slow taper
    "256_128_64_32":   [256, 128, 64, 32],
    "256_128_64_32_16":[256, 128, 64, 32, 16],
}


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:

    # ── 1. Sample hyperparameters ────────────────────────────────────────────
    hidden_layers   = ARCHITECTURES[trial.suggest_categorical("architecture", list(ARCHITECTURES.keys()))]
    epochs          = trial.suggest_int("epochs", 1000, 8000, step=500)
    parameters_lr   = trial.suggest_float("parameters_lr", 1e-5, 1e-2, log=True)

    logger.info(
        f"Trial {trial.number} | hidden_layers={hidden_layers}, "
        f"epochs={epochs}, lr={parameters_lr:.2e}"
    )

    # ── 2. Multi-seed training ────────────────────────────────────────────────
    N_TRAIN_SEEDS = 5   # average over 5 seeds to reduce variance

    trial_costs = []

    for seed in range(N_TRAIN_SEEDS):
        logger.info(f"Trial {trial.number} | seed {seed}/{N_TRAIN_SEEDS - 1} | starting training")

        controller = MultiPeriodNeuralController(
            hidden_layers=hidden_layers,
            activation=ACTIVATION,
            n_periods=2,
        )

        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=100,
            epochs=epochs,
            parameters_lr=parameters_lr,
            log_freq=100,
            seed=seed,
        )

        # Training cost is tracked internally by fit(); retrieve final best
        train_cost = controller.get_average_cost(
            sourcing_model=sourcing_model,
            sourcing_periods=100,
            seed=seed,
        ).item()

        trial_costs.append(train_cost)

        logger.info(
            f"Trial {trial.number} | seed {seed}/{N_TRAIN_SEEDS - 1} | "
            f"train_cost={train_cost:.4f}"
        )


    aggregated = float(torch.tensor(trial_costs).mean().item())
    logger.info(
        f"Trial {trial.number} finished | "
        f"all_seed_costs={[round(c, 4) for c in trial_costs]} | "
        f"aggregated_train_cost={aggregated:.4f}"
    )
    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_TRIALS = 25

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        # pruner=optuna.pruners.NopPruner(),  # no early pruning
        study_name="multi_period_hpo",
        storage="sqlite:///multi_period_hpo.db",  # remove for in-memory only
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"HPO finished | best cost={best.value:.4f} | best params={best.params}")
    print(f"\nBest cost  : {best.value:.4f}")
    print(f"Best params: {best.params}")