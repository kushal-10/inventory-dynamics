from src.idinn.sourcing_model import SingleSourcingModel
from src.idinn.single_controller.base_stock import BaseStockController
from src.idinn.demand import UniformDemand
import logging

logging.basicConfig(
    filename="tests/base_stock.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_base_stock():

    demand_generator = UniformDemand(low=0, high=8)
    model = SingleSourcingModel(lead_time=2,
                                holding_cost=5,
                                shortage_cost=495,
                                init_inventory=6,
                                demand_generator=demand_generator,
                                batch_size=1)

    bs_controller = BaseStockController()
    bs_controller.fit(model)



if __name__ == "__main__":

    test_base_stock()