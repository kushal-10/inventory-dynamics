import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import torch

from ..sourcing_model import DualSourcingModel
from ..dual_controller.capped_dual_index import CappedDualIndexController

# Get root logger
logger = logging.getLogger()


class CyclicCDIController(CappedDualIndexController):

    """
    Only update the restriction, i.e. set q_r at every even period to 0.
    """
    

    def __init__(self, s_e: int = 0, s_r: int = 0, q_r: int = 0) -> None:
        self.sourcing_model = None
        super().__init__(s_e, s_r, q_r)


    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the total cost."""
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for i in range(1, sourcing_periods+1):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            if i%2==0:
                regular_q = 0
                
            sourcing_model.order(regular_q, expedited_q)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
        return total_cost
    

