from src.idinn.periodic_controller.naive_neural_controller import PeriodicNaiveNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

import logging
import torch

logging.basicConfig(
    filename="tests/periodic_controller_tests/naive_nn.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def test_naive_nn_controller():

    model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=UniformDemand(0,4),
        batch_size=1,
    )

    naive_nn_controller = PeriodicNaiveNeuralController()


    naive_nn_controller.fit(
        sourcing_model=model, 
        sourcing_periods=1000,
        epochs=3000,
        log_freq=5,
        init_inventory_freq=6,
    )

    avg_cost = naive_nn_controller.get_periodic_average_cost(sourcing_model=model, sourcing_periods=1000)

    logger.info(f"Average eval cost : {avg_cost}")

if __name__ == "__main__":

    test_naive_nn_controller()