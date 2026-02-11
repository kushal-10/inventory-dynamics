import torch.nn
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.periodic_controller.dynamic_programming import DynamicProgrammingController
from src.idinn.demand import UniformDemand

import logging

logging.basicConfig(
    filename="tests/periodic_controller_tests/dp.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_dynamic_programming():
    demand = UniformDemand(low=0, high=4)

    model = DualSourcingModel(
        regular_lead_time=5,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=demand,
        batch_size=1
    )

    dp_controller = DynamicProgrammingController()
    dp_controller.fit(
        sourcing_model=model,
    )

if __name__ == "__main__":
    test_dynamic_programming()