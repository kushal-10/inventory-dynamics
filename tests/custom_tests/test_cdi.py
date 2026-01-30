from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.dual_controller.capped_dual_index import CappedDualIndexController
from src.idinn.demand import UniformDemand

import logging

logging.basicConfig(
    filename="tests/custom_tests/cdi.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_capped_dual_index():

    demand = UniformDemand(low=0, high=4)

    model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=demand,
        batch_size=1
    )

    cdi_controller = CappedDualIndexController()

    cdi_controller.fit(sourcing_model=model, sourcing_periods=1000)


if __name__ == "__main__":
    test_capped_dual_index()