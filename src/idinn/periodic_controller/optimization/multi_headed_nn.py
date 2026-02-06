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

        head_even_layers = trial.suggest_categorical(
            "head_even_layers",
            [
                [8, 4],
                [4],
                [8]
            ],
        )

        head_odd_layers = trial.suggest_categorical(
            "head_odd_layers",
            [
                [8, 4],
                [4],
                [8]
            ],
        )

        MAX_Q = trial.suggest_categorical(
            "MAX_Q",
            [
                i for i in range(17,24)
            ]
        )

        parameters_lr_shared = trial.suggest_loguniform("parameters_lr_shared", 1e-4, 1e-2)
        parameters_lr_even = trial.suggest_loguniform("parameters_lr_even", 1e-4, 1e-2)
        parameters_lr_odd = trial.suggest_loguniform("parameters_lr_odd", 1e-4, 1e-2)

        weight_decay_shared = trial.suggest_loguniform("weight_decay_shared", 1e-6, 1e-3)
        weight_decay_even = trial.suggest_loguniform("weight_decay_even", 1e-6, 1e-3)
        weight_decay_odd = trial.suggest_loguniform("weight_decay_odd", 1e-6, 1e-3)


        mlflow.log_params({
            "shared_layers": str(shared_layers),
            "head_even_layers": str(head_even_layers),
            "head_odd_layers": str(head_odd_layers),
            "parameters_lr_shared": parameters_lr_shared,
            "weight_decay_shared": weight_decay_shared,
            "parameters_lr_odd": parameters_lr_odd,
            "parameters_lr_even": parameters_lr_even,
            "weight_decay_even": weight_decay_even,
            "weight_decay_odd": weight_decay_odd,
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
            head_even_layers=head_even_layers,
            head_odd_layers=head_odd_layers,
            MAX_Q=MAX_Q
        )

        controller.fit(
            sourcing_model=sourcing_model,
            sourcing_periods=100,
            epochs=600,
            weight_decay_shared = weight_decay_shared,
            weight_decay_odd = weight_decay_odd,
            weight_decay_even = weight_decay_even,
            parameters_lr_shared = parameters_lr_shared,
            parameters_lr_odd = parameters_lr_odd,
            parameters_lr_even = parameters_lr_even,
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


"""
[I 2026-02-05 13:27:49,383] Trial 13 finished with value: 26.93000030517578 and parameters: {'shared_layers': [64, 32, 16], 
'head_even_layers': [8, 4], 
'head_odd_layers': [4], 
'MAX_Q': 20, 
'parameters_lr_shared': 0.004284520099077406, 
'parameters_lr_even': 0.000372801186818003, 
'parameters_lr_odd': 0.00034988574403358624, 
'weight_decay_shared': 0.000282468537229603, 
'weight_decay_even': 1.1267508218370485e-06, 
'weight_decay_odd': 5.343701588766947e-06

}. Best is trial 13 with value: 26.93000030517578.
"""