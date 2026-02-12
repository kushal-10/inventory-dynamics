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
    filename="tests/periodic_controller_tests/mhnn.log",
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

    """
    shared_layers': [64, 32], 
    'head_restricted_layers': [8], 
    'head_regular_layers': [4], 
    'MAX_Q': 19, 
    'parameters_lr_shared': 0.008699514378438841, 
    'parameters_lr_restricted': 0.0017120492075871011, 
    'parameters_lr_regular': 0.00025845752177043255, 
    """
    
    controller = MultiHeadedNeuralController(
        shared_layers=[256, 128, 64, 32, 16],
        head_regular_layers=[8],
        head_restricted_layers=[8],
        MAX_Q=15,
    )

    """
    Trial 24 finished with value: 38031.72265625 and parameters: 
    {'shared_layers': [256, 128, 64, 32, 16], 'head_restricted_layers': [8], 'head_regular_layers': [8], 'MAX_Q': 15, 
    'parameters_lr_shared': 0.00119973482423591, 
    'parameters_lr_restricted': 0.00029337351258512255, 
    'parameters_lr_regular': 0.000547768302500623, 'weight_decay_shared': 2.716712467630229e-06, 
    'weight_decay_restricted': 1.88618464952265e-05, 'weight_decay_regular': 0.0002540269356565944}. Best is trial 5 with value: 32.060001373291016.
    """
    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=3000,
        weight_decay_shared = 2.716712467630229e-06,
        weight_decay_regular = 0.0002540269356565944,
        weight_decay_restricted = 1.88618464952265e-05,
        parameters_lr_shared = 0.00119973482423591,
        parameters_lr_regular = 0.000547768302500623,
        parameters_lr_restricted = 0.00029337351258512255,
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

