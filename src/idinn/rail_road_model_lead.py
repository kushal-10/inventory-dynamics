from typing import Optional, Union

import torch

from .demand import CustomDemand, UniformDemand
from .demand_three_subperiods import DiscreteTruncatedGammaDemand


class RailRoadInventoryModelLeadTime:
    """
    Periodic rail-road inventory model with LEAD TIME.

    - Time is discrete (daily)
    - Demand occurs every day
    - Orders arrive after a fixed lead time L
    - Regular (rail) orders allowed only every K days
    - Expedited (road) orders allowed on all other days
    - Single-item, single-location
    """

    def __init__(
        self,
        cycle_length: int,
        lead_time: int,
        regular_order_cost: float,
        expedited_order_cost: float,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator: Union[
            UniformDemand, CustomDemand, DiscreteTruncatedGammaDemand
        ],
        batch_size: int = 1,
    ):
        if cycle_length <= 0:
            raise ValueError("`cycle_length` must be positive.")
        if lead_time < 0:
            raise ValueError("`lead_time` must be non-negative.")

        self.cycle_length = cycle_length
        self.lead_time = lead_time

        self.regular_order_cost = regular_order_cost
        self.expedited_order_cost = expedited_order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

        self.demand_generator = demand_generator
        self.batch_size = batch_size

        self.init_inventory = torch.tensor(
            [init_inventory], dtype=torch.float, requires_grad=False
        )

        self.reset()

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self.batch_size = batch_size

        # inventory history
        self.past_inventories = self.get_init_inventory().repeat(self.batch_size, 1)

        # demand history
        self.past_demands = torch.zeros(self.batch_size, 1)

        # order history
        self.past_orders = torch.zeros(self.batch_size, 1)

        # outstanding pipeline (q₁,...,q_L)
        if self.lead_time > 0:
            self.pipeline = torch.zeros(self.batch_size, self.lead_time)
        else:
            self.pipeline = torch.zeros(self.batch_size, 0)

        # cycle phase τ
        self.current_cycle_day = 0

    def get_init_inventory(self) -> torch.Tensor:
        # force integer inventory
        return self.init_inventory - torch.frac(self.init_inventory).detach()

    def get_current_inventory(self) -> torch.Tensor:
        return self.past_inventories[:, [-1]]

    def get_current_cycle_day(self) -> int:
        return self.current_cycle_day

    def get_pipeline(self):
        """
        Returns pipeline as Python list (DP-friendly).
        Assumes batch_size = 1.
        """
        return self.pipeline[0].int().tolist()

    # ------------------------------------------------------------------
    # Cost accessors
    # ------------------------------------------------------------------

    def get_holding_cost(self) -> float:
        return self.holding_cost

    def get_shortage_cost(self) -> float:
        return self.shortage_cost

    def get_regular_order_cost(self) -> float:
        return self.regular_order_cost

    def get_expedited_order_cost(self) -> float:
        return self.expedited_order_cost

    # ------------------------------------------------------------------
    # Ordering logic
    # ------------------------------------------------------------------

    def is_regular_day(self) -> bool:
        return self.current_cycle_day == 0

    def order(self, q: torch.Tensor, seed: Optional[int] = None) -> None:
        """
        Place an order and advance the system by one period.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor([[q]])

        # record order
        self.past_orders = torch.cat([self.past_orders, q], dim=1)

        # sample demand
        current_demand = self.demand_generator.sample(self.batch_size)
        self.past_demands = torch.cat([self.past_demands, current_demand], dim=1)

        # arrivals from pipeline
        if self.lead_time > 0:
            arrived = self.pipeline[:, [0]]

            # shift pipeline and append new order
            self.pipeline = torch.cat(
                [self.pipeline[:, 1:], q], dim=1
            )
        else:
            arrived = q

        # inventory update
        current_inventory = (
            self.get_current_inventory() + arrived - current_demand
        )

        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )

        # advance cycle
        self.current_cycle_day = (self.current_cycle_day + 1) % self.cycle_length