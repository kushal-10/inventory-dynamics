import logging
import torch
from tqdm import tqdm 

from src.idinn.periodic_controller.multi_headed_neural_network import MultiHeadedNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/periodic_controller/results/multi_headed.log",
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
    
    controller = MultiHeadedNeuralController(
        shared_layers=[64,32,16],
        head_even_layers=[8,4],
        head_odd_layers=[4]
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=3000,
        weight_decay_shared = 9.563411263437158e-05,
        weight_decay_odd = 0.0007149281582778022,
        weight_decay_even = 1.2386176014952e-06,
        parameters_lr_shared = 0.0013542080013067383,
        parameters_lr_odd = 0.0008940175735593203,
        parameters_lr_even = 0.0012104440658971336,
        seed=42,
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
        for seed in tqdm(range(500)):
            cost = controller.get_periodic_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
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