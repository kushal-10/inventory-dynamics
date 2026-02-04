
from typing import Optional, no_type_check

import torch

from ..sourcing_model import DualSourcingModel
from ..dual_controller.base import BaseDualController

class BasePeriodicDualController(BaseDualController):

    """
    Only update the cost calculation to include a phase argument.
    """

    @no_type_check
    def get_periodic_total_cost(
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
            regular_q, expedited_q = self.predict( #cleaned
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                phase=i%2, # Phase starts from 1 (odd time)
                output_tensor=True,
            )
            sourcing_model.order(regular_q, expedited_q) #cleaned
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
        return total_cost

    @no_type_check
    def get_periodic_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average cost."""
        return (
            self.get_periodic_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )
