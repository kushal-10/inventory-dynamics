from src.idinn.sourcing_model import SingleSourcingModel
from src.idinn.single_controller.base_stock import BaseStockController
from src.idinn.single_controller.single_neural import SingleSourcingNeuralController
from src.idinn.demand import UniformDemand
import logging

logging.basicConfig(
    filename="tests/custom_tests/single_neural.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_neural():
    demand_generator = UniformDemand(low=0, high=4)
    model = SingleSourcingModel(lead_time=2,
                                holding_cost=5,
                                shortage_cost=495,
                                init_inventory=6,
                                demand_generator=demand_generator,
                                batch_size=1)

    sn_controller = SingleSourcingNeuralController()
    sn_controller.fit(
        sourcing_model=model,
        sourcing_periods=5,
        validation_sourcing_periods=1,
        epochs=2,
        seed=1,)

    sn_controller.predict(current_inventory=6,)


if __name__ == "__main__":
    test_neural()