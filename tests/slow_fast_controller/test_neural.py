from src.idinn.slow_fast import CyclicSlowFastModel
from src.idinn.slow_fast_controller.slow_fast_neural import CyclicSlowFastNeuralController
from src.idinn.slow_fast_controller.base import BaseSlowFastController
from src.idinn.demand import UniformDemand

import torch
import logging

logging.basicConfig(
    filename="tests/slow_fast_controller/neural.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def test_neural():
    model = CyclicSlowFastModel(
        cycle=2,
        slow_lead_time=2,
        fast_lead_time=0,
        slow_order_cost=0,
        fast_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=UniformDemand(low=0, high=4),
        batch_size=1,
    )

    sf_neural_controller = CyclicSlowFastNeuralController()
    sf_neural_controller.fit(
        sourcing_model=model,
        sourcing_periods=100,
        epochs=100,
    )

    pass

if __name__ == "__main__":
    test_neural()

