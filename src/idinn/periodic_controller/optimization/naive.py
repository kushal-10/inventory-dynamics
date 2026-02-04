import logging
import torch
import optuna
import mlflow

from src.idinn.periodic_controller.naive_neural_controller import (
    PeriodicNaiveNeuralController
)
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/periodic_controller/optimization/naive_nn_hpo.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------
def objective(trial):

    with mlflow.start_run(nested=True):

        # ------------------ search space ------------------
        hidden_layers = trial.suggest_categorical(
            "hidden_layers",
            [
                [32, 16],
                [64, 32],
                [128, 32, 8],
                [128, 32, 8, 4],
            ],
        )

        parameters_lr = trial.suggest_loguniform("parameters_lr", 1e-4, 5e-3)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

        # ------------------ log params ------------------
        mlflow.log_params({
            "hidden_layers": str(hidden_layers),
            "parameters_lr": parameters_lr,
            "weight_decay": weight_decay,
        })

        # ------------------ model ------------------
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

        controller = PeriodicNaiveNeuralController(
            hidden_layers=hidden_layers
        )

        # ------------------ training (cheap) ------------------
        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=100,
            epochs=800,
            parameters_lr=parameters_lr,
            weight_decay=weight_decay,
            seed=42,
        )

        # ------------------ evaluation ------------------
        with torch.no_grad():
            avg_cost = controller.get_periodic_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
                seed=123,
            )

        mlflow.log_metric("eval_cost", avg_cost.item())

        return avg_cost.item()


# ---------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------
def run_hpo():

    mlflow.set_experiment("PeriodicNaiveNN-HPO")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    logger.info(f"Best cost: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


# ---------------------------------------------------------------------
# final training + evaluation
# ---------------------------------------------------------------------
def train_final_model(best_params):

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

    controller = PeriodicNaiveNeuralController(
        hidden_layers=best_params["hidden_layers"]
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=300,
        epochs=2000,
        parameters_lr=best_params["parameters_lr"],
        weight_decay=best_params["weight_decay"],
        seed=42,
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
        for seed in range(50):
            cost = controller.get_periodic_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=2000,
                seed=seed,
            )
            costs.append(cost)

    mean_cost = torch.mean(torch.stack(costs))
    std_cost = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost}")
    logger.info(f"Final std cost: {std_cost}")

    print("Final mean:", mean_cost.item())
    print("Final std:", std_cost.item())


# ---------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":

    best_params = run_hpo()
    train_final_model(best_params)
