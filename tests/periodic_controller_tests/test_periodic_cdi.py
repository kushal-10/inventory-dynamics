from src.idinn.periodic_controller.periodic_cdi_controller import PeriodicCDIController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

import logging

logging.basicConfig(
    filename="tests/periodic_controller_tests/periodic_cdi.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_cyc_cdi():

    model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=UniformDemand(0,4),
        batch_size=1
    )

    cyc_cdi_controller = PeriodicCDIController()

    cyc_cdi_controller.fit(sourcing_model=model, sourcing_periods=1000)

    cyc_cdi_controller.get_average_cost(sourcing_model=model, sourcing_periods=1000)

if __name__ == "__main__":

    test_cyc_cdi()