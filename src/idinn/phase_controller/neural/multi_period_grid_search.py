import logging
from typing import Dict, List

import torch
import optuna
from optuna.samplers import TPESampler

from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/phase_controller/neural/multi_period_optuna.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress optuna's internal logs if desired
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

ACTIVATION_MAP = {
    "celu": torch.nn.CELU(alpha=1.0),
    "relu": torch.nn.ReLU(),
    "elu":  torch.nn.ELU(),
    "silu": torch.nn.SiLU(),
    "tanh": torch.nn.Tanh(),
}

def build_hidden_layers(trial: optuna.Trial) -> List[int]:
    """Sample architecture: depth then per-layer width."""
    n_layers = trial.suggest_int("n_layers", 2, 6)
    hidden_layers = []
    for i in range(n_layers):
        size = trial.suggest_categorical(f"layer_{i}_size", [8, 16, 32, 64, 128, 256])
        hidden_layers.append(size)
    return hidden_layers


def objective(trial: optuna.Trial) -> float:
    # --- sample hyperparameters ---
    hidden_layers = build_hidden_layers(trial)
    parameters_lr = trial.suggest_float("parameters_lr", 1e-5, 1e-2, log=True)
    epochs        = trial.suggest_int("epochs", 1000, 8000, step=500)
    activation    = ACTIVATION_MAP[trial.suggest_categorical("activation", list(ACTIVATION_MAP.keys()))]

    logger.info(
        f"Trial {trial.number} | hidden_layers={hidden_layers}, "
        f"lr={parameters_lr:.2e}, epochs={epochs}, "
        f"activation={trial.params['activation']}"
    )

    # --- multi-seed training + eval (same protocol as your grid search) ---
    N_TRAIN_SEEDS = 5   # keep cheap; increase for final rerun
    N_EVAL_SEEDS  = 50

    trial_costs = []
    for seed in range(N_TRAIN_SEEDS):
        controller = MultiPeriodNeuralController(
            hidden_layers=hidden_layers,
            activation=activation,
            n_periods=2,
        )
        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=10,
            epochs=epochs,
            parameters_lr=parameters_lr,
            seed=seed,
        )

        costs = []
        with torch.no_grad():
            for eval_seed in range(N_EVAL_SEEDS):
                cost = controller.get_average_cost(
                    sourcing_model=sourcing_model,
                    sourcing_periods=1000,
                    seed=eval_seed,
                )
                costs.append(cost)

        mean_cost = torch.mean(torch.stack(costs)).item()
        trial_costs.append(mean_cost)

        # Optuna pruning: report intermediate value after each seed
        trial.report(mean_cost, step=seed)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    aggregated = float(torch.tensor(trial_costs).mean().item())
    logger.info(f"Trial {trial.number} finished | mean_cost={aggregated:.4f}")
    return aggregated


if __name__ == "__main__":
    N_TRIALS = 100

    study = optuna.create_study(
        direction="minimize", 
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name="multi_period_hpo",
        storage="sqlite:///multi_period_hpo.db",   # persists results; remove for in-memory
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    logger.info(f"HPO finished | best cost={best.value:.4f} | best params={best.params}")
    print(f"\nBest cost : {best.value:.4f}")
    print(f"Best params: {best.params}")