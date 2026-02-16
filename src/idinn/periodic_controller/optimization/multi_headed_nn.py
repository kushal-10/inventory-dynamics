import logging
import torch
import optuna
import mlflow

from src.idinn.periodic_controller.multi_headed_neural_network import MultiHeadedNeuralController

from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/periodic_controller/optimization/multi_headed.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def objective(trial):

    with mlflow.start_run():
        
        shared_layers = trial.suggest_categorical(
            "shared_layers",
            [
                [128, 64, 32, 16, 8],
                [64, 32, 16, 8],
                [32, 16],
                [32, 16, 8],
                [16, 8],
                [64, 32],
                [64, 32, 16],
            ],
        )

        head_restricted_layers = trial.suggest_categorical(
            "head_restricted_layers",
            [
                [8, 4],
                [4],
                [8]
            ],
        )

        head_regular_layers = trial.suggest_categorical(
            "head_regular_layers",
            [
                [8, 4],
                [4],
                [8]
            ],
        )

        MAX_Q = trial.suggest_categorical(
            "MAX_Q",
            [
                i for i in range(10,30)
            ]
        )

        parameters_lr_shared = trial.suggest_loguniform("parameters_lr_shared", 1e-4, 1e-2)
        parameters_lr_restricted = trial.suggest_loguniform("parameters_lr_restricted", 1e-4, 1e-2)
        parameters_lr_regular = trial.suggest_loguniform("parameters_lr_regular", 1e-4, 1e-2)

        weight_decay_shared = trial.suggest_loguniform("weight_decay_shared", 1e-6, 1e-3)
        weight_decay_restricted = trial.suggest_loguniform("weight_decay_restricted", 1e-6, 1e-3)
        weight_decay_regular = trial.suggest_loguniform("weight_decay_regular", 1e-6, 1e-3)


        mlflow.log_params({
            "shared_layers": str(shared_layers),
            "head_regular_layers": str(head_regular_layers),
            "head_restricted_layers": str(head_restricted_layers),
            "parameters_lr_shared": parameters_lr_shared,
            "weight_decay_shared": weight_decay_shared,
            "parameters_lr_restricted": parameters_lr_restricted,
            "parameters_lr_regular": parameters_lr_regular,
            "weight_decay_restricted": weight_decay_restricted,
            "weight_decay_regular": weight_decay_regular,
            "MAX_Q": MAX_Q
        })

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

        controller = MultiHeadedNeuralController(
            shared_layers=shared_layers,
            head_regular_layers=head_regular_layers,
            head_restricted_layers=head_restricted_layers,
            MAX_Q=MAX_Q
        )

        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=100,
            epochs=600,
            weight_decay_shared = weight_decay_shared,
            weight_decay_restricted = weight_decay_restricted,
            weight_decay_regular = weight_decay_regular,
            parameters_lr_shared = parameters_lr_shared,
            parameters_lr_restricted = parameters_lr_restricted,
            parameters_lr_regular = parameters_lr_regular,
            seed=42,
        )

        with torch.no_grad():
            avg_cost = controller.get_periodic_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
                seed=123,
            )

        mlflow.log_metric("eval_cost", avg_cost.item())

        return avg_cost.item()


# ---------------------------------------------------------------------
def run_hpo():

    mlflow.set_experiment("PeriodicNaiveNN-HPO")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    logger.info(f"Best cost: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


if __name__ == '__main__':
    best_params = run_hpo()
    logger.info(f"Best Parameters: {best_params}")

