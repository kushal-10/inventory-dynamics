import logging
import torch
from tqdm import tqdm 

from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="tests/phase_controller/test_multi_period.log",
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

def test_multi_period_model():
    
    controller = MultiPeriodNeuralController(
        hidden_layers=[64, 32, 16, 8],
        n_periods=2
    )

    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=10000,
        validation_sourcing_periods=1000,
        parameters_lr=3e-4,
        # init_inventory_lr=1e-4,
        seed=42,
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
        for seed in tqdm(range(50)):
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

    print("Final mean:", mean_cost.item())
    print("Final std:", std_cost.item())



if __name__ == '__main__':
    test_multi_period_model()
