import torch.nn
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.dual_controller.dual_neural import DualSourcingNeuralController
from src.idinn.demand import UniformDemand

import logging

logging.basicConfig(
    filename="tests/custom_tests/dual_neural.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def test_dual():
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
        batch_size=512
    )

    dnn_controller = DualSourcingNeuralController(
        activation=torch.nn.CELU(alpha=0.99)
    )

    dnn_controller.fit(sourcing_model=model,
                       sourcing_periods=100,
                       epochs=500,)

    avg = dnn_controller.get_average_cost(
        sourcing_model=model,
        sourcing_periods=1000,
    )

    logger.info(f"Average cost: {avg}")


if __name__ == "__main__":
    test_dual()