"""
FIX
"""

import torch
from functools import lru_cache
from typing import Optional, Tuple, Union, List
from tqdm import tqdm

from .base import BaseSlowFastController
from ..slow_fast import CyclicSlowFastModel


class DynamicProgrammingController(BaseSlowFastController):
    """
    Rolling-horizon Dynamic Programming controller
    for CyclicSlowFastModel.
    """

    def __init__(
        self,
        horizon: int = 3,
        I_min: int = -20,
        I_max: int = 20,
        Q_max: int = 20,
    ):
        self.horizon = horizon
        self.I_min = I_min
        self.I_max = I_max
        self.Q_max = Q_max
        self.sourcing_model: Optional[CyclicSlowFastModel] = None

    def fit(self, sourcing_model: CyclicSlowFastModel, **kwargs) -> None:
        self.sourcing_model = sourcing_model

    @lru_cache(None)
    def _bellman(self, t, inventory, slow_pipeline, phase):
        if t == self.horizon:
            return 0.0

        model = self.sourcing_model
        h = model.get_holding_cost()
        b = model.get_shortage_cost()

        best_cost = float("inf")

        # action sets
        slow_actions = range(self.Q_max + 1) if phase == 0 else [0]
        fast_actions = range(self.Q_max + 1)

        for q_s in slow_actions:
            for q_f in fast_actions:

                expected_cost = 0.0

                for d in model.demand_generator.support():
                    p = model.demand_generator.prob(d)

                    # arrivals
                    arrived_slow = slow_pipeline[0] if slow_pipeline else 0

                    # inventory evolution (order → arrival → demand)
                    next_inventory = inventory + arrived_slow + q_f - d

                    next_inventory = max(self.I_min, min(self.I_max, next_inventory))

                    # pipeline update
                    next_pipeline = slow_pipeline[1:] + (q_s,)
                    next_phase = (phase + 1) % model.cycle

                    holding = h * max(next_inventory, 0)
                    shortage = b * max(-next_inventory, 0)

                    future = self._bellman(
                        t + 1,
                        next_inventory,
                        next_pipeline,
                        next_phase,
                    )

                    expected_cost += p * (holding + shortage + future)

                best_cost = min(best_cost, expected_cost)

        return best_cost

    def predict(
            self,
            current_inventory: Union[int, torch.Tensor],
            past_slow_orders: Optional[Union[List[int], torch.Tensor]] = None,
            past_fast_orders=None,
            output_tensor: bool = False,
    ):

        if self.sourcing_model is None:
            raise AttributeError("Controller is not fitted.")

        model = self.sourcing_model

        I = int(self._check_current_inventory(current_inventory).item())

        slow_lt = model.get_slow_lead_time()
        pipeline = self._check_past_orders(past_slow_orders, slow_lt)
        pipeline_tuple = tuple(int(x) for x in pipeline[0].tolist())

        phase = model.get_cycle_phase()

        best_action = None
        best_cost = float("inf")

        slow_actions = range(self.Q_max + 1) if phase == 0 else [0]
        fast_actions = range(self.Q_max + 1)

        for q_s in slow_actions:
            for q_f in fast_actions:

                cost = (
                        model.get_slow_order_cost() * q_s
                        + model.get_fast_order_cost() * q_f
                        + self._bellman(
                    0,  # start Bellman at current time
                    I,  # current inventory
                    pipeline_tuple,  # current pipeline
                    phase,  # current phase
                )
                )

                if cost < best_cost:
                    best_cost = cost
                    best_action = (q_s, q_f)

        q_s, q_f = best_action

        if output_tensor:
            return torch.tensor([[q_s]]), torch.tensor([[q_f]])

        return q_s, q_f

    def reset(self) -> None:
        self.sourcing_model = None
        self._bellman.cache_clear()

