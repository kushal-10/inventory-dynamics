from src.idinn.slow_fast import CyclicSlowFastModel
from src.idinn.slow_fast_controller.dynamic_programming import (
    DynamicProgrammingController,
)
from src.idinn.demand import UniformDemand

import logging

logging.basicConfig(
    filename="tests/slow_fast_controller/dp.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def evaluate_controller(name, controller, model, T=500, seed=42):
    model.reset()
    controller.fit(model)

    avg_cost = controller.get_average_cost(model, T, seed).item()
    inventories = model.get_past_inventories()

    min_inventory = inventories.min().item()
    backlog_periods = (inventories < 0).sum().item()
    inventory_var = inventories.var().item()

    logger.info(
        f"{name} | "
        f"avg_cost={avg_cost:.2f}, "
        f"min_inventory={min_inventory:.2f}, "
        f"backlog_periods={backlog_periods}, "
        f"inventory_var={inventory_var:.2f}"
    )

    print(
        f"{name:20s} | "
        f"avg_cost={avg_cost:10.2f} | "
        f"min_I={min_inventory:7.2f} | "
        f"backlog={backlog_periods:5d} | "
        f"var={inventory_var:8.2f}"
    )


def test_dp():
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

    dp_controller = DynamicProgrammingController(
        horizon=3,     # try 2, 3, 4
        I_min=-20,
        I_max=20,
        Q_max=10,
    )

    evaluate_controller("RollingDP", dp_controller, model)


if __name__ == "__main__":
    test_dp()
