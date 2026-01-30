import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import torch
from tqdm import tqdm

from ..slow_fast import CyclicSlowFastModel
from .base import BaseSlowFastController

logger = logging.getLogger()


class TripleIndexController(BaseSlowFastController):
    """
    Phase-dependent triple-index controller for cyclic slow–fast sourcing.
    """

    def __init__(self, Sr_0: int = 0, Se_0: int = 0, Se_1: int = 0) -> None:
        self.sourcing_model: Optional[CyclicSlowFastModel] = None
        self.Sr_0 = Sr_0
        self.Se_0 = Se_0
        self.Se_1 = Se_1
        logger.info("Initialized TripleIndexController")

    # -------------------------------------------------

    def _effective_inventory_position(
        self,
        current_inventory: torch.Tensor,
        past_slow_orders: torch.Tensor,
        slow_lead_time: int,
    ) -> torch.Tensor:
        if slow_lead_time > 0:
            return current_inventory + past_slow_orders[:, -slow_lead_time:].sum(
                dim=1, keepdim=True
            )
        return current_inventory

    # -------------------------------------------------

    @no_type_check
    def fit(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        Sr_0_range: torch.Tensor = torch.arange(2, 11),
        Se_0_range: torch.Tensor = torch.arange(2, 11),
        Se_1_range: torch.Tensor = torch.arange(2, 11),
        seed: Optional[int] = None,
    ) -> None:

        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        min_cost = torch.inf
        start_time = datetime.now()
        logger.info(f"Starting triple-index grid search at {start_time}")

        for Sr_0 in tqdm(Sr_0_range):
            for Se_0 in Se_0_range:
                for Se_1 in Se_1_range:
                    sourcing_model.reset()
                    self.Sr_0 = Sr_0
                    self.Se_0 = Se_0
                    self.Se_1 = Se_1

                    total_cost = self.get_total_cost(
                        sourcing_model, sourcing_periods
                    )

                    logger.info(f"Triple-index grid search for Sr_0 = {Sr_0}, Se_0 = {Se_0}, Se_1 = {Se_1} resulted in cost = {total_cost/sourcing_periods:.2f}")
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best = (Sr_0, Se_0, Se_1)

        self.Sr_0, self.Se_0, self.Se_1 = best
        logger.info(
            f"Optimal parameters: Sr_0={best[0]}, Se_0={best[1]}, Se_1={best[2]} for cost = {min_cost/sourcing_periods:.2f}"
        )

    # -------------------------------------------------

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_slow_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_fast_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:

        if self.sourcing_model is None:
            raise AttributeError("Controller is not trained.")

        model = self.sourcing_model
        slow_lt = model.get_slow_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_slow_orders = self._check_past_orders(past_slow_orders, slow_lt)

        phase = model.get_cycle_phase()

        eip = self._effective_inventory_position(
            current_inventory, past_slow_orders, slow_lt
        )

        if phase == 0:
            slow_q = torch.clamp(self.Sr_0 - eip, min=0)
            fast_q = torch.clamp(self.Se_0 - eip, min=0)
        else:
            slow_q = torch.zeros_like(eip)
            fast_q = torch.clamp(self.Se_1 - eip, min=0)

        if output_tensor:
            return slow_q, fast_q
        return int(slow_q.item()), int(fast_q.item())

    # -------------------------------------------------

    def reset(self) -> None:
        self.Sr_0 = 0
        self.Se_0 = 0
        self.Se_1 = 0
        self.sourcing_model = None
