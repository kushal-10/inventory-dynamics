from src.idinn.slow_fast import CyclicSlowFastModel
from src.idinn.slow_fast_controller.triple_index import TripleIndexController
from src.idinn.slow_fast_controller.base import BaseSlowFastController
from src.idinn.demand import UniformDemand

import torch
import logging

logging.basicConfig(
    filename="tests/slow_fast_controller/triple_index.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


"""
Baseline Classes
"""
class AlwaysExpediteController(BaseSlowFastController):
    def fit(self, sourcing_model, **kwargs):
        self.sourcing_model = sourcing_model

    def predict(
        self,
        current_inventory,
        past_slow_orders=None,
        past_fast_orders=None,
        output_tensor=False,
    ):
        I = self._check_current_inventory(current_inventory)
        slow_q = torch.zeros_like(I)
        fast_q = torch.clamp(-I, min=0)

        if output_tensor:
            return slow_q, fast_q
        return int(slow_q.item()), int(fast_q.item())

    def reset(self):
        self.sourcing_model = None

class SingleBaseStockController(BaseSlowFastController):
    def __init__(self, S: int):
        self.S = S

    def fit(self, sourcing_model, **kwargs):
        self.sourcing_model = sourcing_model

    def predict(
        self,
        current_inventory,
        past_slow_orders=None,
        past_fast_orders=None,
        output_tensor=False,
    ):
        I = self._check_current_inventory(current_inventory)
        slow_q = torch.clamp(self.S - I, min=0)
        fast_q = torch.zeros_like(I)

        if output_tensor:
            return slow_q, fast_q
        return int(slow_q.item()), int(fast_q.item())

    def reset(self):
        self.sourcing_model = None



def evaluate_controller(name, controller, model, T=1000, seed=42):
    model.reset()

    if name=="TripleIndex":
        controller.fit(model, sourcing_periods=T)
    else:
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


def test_index():
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

    controllers = {
        "TripleIndex": TripleIndexController(),
        # "AlwaysExpedite": AlwaysExpediteController(),
        # "SingleBaseStock(S=10)": SingleBaseStockController(S=10),
    }

    for name, ctrl in controllers.items():
        evaluate_controller(name, ctrl, model)


if __name__ == "__main__":
    test_index()