import logging
from typing import Dict

import torch
from tqdm import tqdm 

from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/phase_controller/neural/multi_period_grid_search.log",
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

def test_multi_period_model(params: Dict = None):
    
    hidden_layers = params["hidden_layers"]
    seed = params["seed"]
    init_inv_freq = params["init_inv_freq"]
    inventory_lr = params["inventory_lr"]
    parameters_lr = params["parameters_lr"]


    controller = MultiPeriodNeuralController(
        hidden_layers=hidden_layers,
        n_periods=2
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=10,
        epochs=5000,
        init_inventory_freq=init_inv_freq,
        init_inventory_lr=inventory_lr,
        parameters_lr=parameters_lr,
        seed=seed
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
        for seed in range(50):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
                seed=seed,
            )
            costs.append(cost)

    mean_cost = torch.mean(torch.stack(costs))
    std_cost = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost}")
    logger.info(f"Final std cost: {std_cost}")

    return mean_cost



if __name__ == '__main__':

    seeds = list(range(50))
    layers = [
        # [256, 128, 64, 32, 16, 8],
        [128, 64, 32, 16, 8],
        # [64, 32, 16, 8],
        # [32, 16, 8]
    ]
    inv_freqs = [4,]
    inv_lrs = [1e-1]
    param_lrs = [3e-3]

    configs = []
    for seed in seeds:
        for layer in layers:
            for inv_freq in inv_freqs:
                for inv_lr in inv_lrs:
                    for param_lr in param_lrs:
                        config_dict = {
                            "hidden_layers": layer,
                            "seed": seed,
                            "init_inv_freq": inv_freq,
                            "inventory_lr": inv_lr,
                            "parameters_lr": param_lr
                        }
                        configs.append(config_dict)

    OPT_COST = 1000
    OPT_PARAMS = None
    for c in tqdm(configs):
        logger.info(f"Starting training for config : {c}")
        mean_cost = test_multi_period_model(params=c)
        if mean_cost <= OPT_COST:
            OPT_COST = mean_cost
            OPT_PARAMS = c
    logger.info(f"HPO search finished with minimum cost - {OPT_COST} with optimal parameters : {OPT_PARAMS}")
    