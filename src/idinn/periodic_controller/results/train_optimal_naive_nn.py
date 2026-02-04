import logging
import torch

from src.idinn.periodic_controller.naive_neural_controller import PeriodicNaiveNeuralController

from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/periodic_controller/results/naive_nn.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


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

def train_best_model():
    best_params = {
        "hidden_layers": [32, 16],
        "parameters_lr": 0.0046258,
        "weight_decay":9.38279701e-6,
    }
    controller = PeriodicNaiveNeuralController(
        hidden_layers=best_params["hidden_layers"]
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=500,
        epochs=3000,
        parameters_lr=best_params["parameters_lr"],
        weight_decay=best_params["weight_decay"],
        seed=42,
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
        for seed in range(500):
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



if __name__ == '__main__':
    train_best_model()