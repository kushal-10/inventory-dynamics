from abc import abstractmethod
from typing import Optional, Union
import torch
from .demand import CustomDemand, UniformDemand

from .sourcing_model import BaseSourcingModel

class BaseSlowFastModel(BaseSourcingModel):
    def __init__(
        self,
        slow_lead_time: int,
        fast_lead_time: int,
        slow_order_cost: float,
        fast_order_cost: float,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator,
        batch_size: int = 1,
    ):
        if slow_lead_time < fast_lead_time:
            raise ValueError("slow_lead_time must be >= fast_lead_time")

        super().__init__(
            regular_lead_time=slow_lead_time,
            expedited_lead_time=fast_lead_time,
            regular_order_cost=slow_order_cost,
            expedited_order_cost=fast_order_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            init_inventory=init_inventory,
            batch_size=batch_size,
            demand_generator=demand_generator,
        )

    # ---------- semantic helpers ----------
    def get_slow_lead_time(self) -> int:
        return self.regular_lead_time

    def get_fast_lead_time(self) -> int:
        return self.expedited_lead_time

    def get_slow_order_cost(self) -> float:
        return self.regular_order_cost

    def get_fast_order_cost(self) -> float:
        return self.expedited_order_cost

    # ---------- pipeline helpers ----------
    def get_slow_pipeline(self) -> torch.Tensor:
        """All outstanding slow orders"""
        return self.past_regular_orders

    def get_fast_pipeline(self) -> torch.Tensor:
        """All outstanding fast orders"""
        return self.past_expedited_orders

    def get_last_slow_order(self) -> torch.Tensor:
        """
        Quantity of the most recent slow order (q^s_t).
        Shape: (batch_size, 1)
        """
        return self.past_regular_orders[:, [-1]]

    def get_last_fast_order(self) -> torch.Tensor:
        """
        Quantity of the most recent fast order (q^f_t).
        Shape: (batch_size, 1)
        """
        return self.past_expedited_orders[:, [-1]]

    # ---------- inventory helpers ----------
    def get_effective_inventory_position(self) -> torch.Tensor:
        """
        Standard inventory position proxy:
        I_t + sum of slow pipeline
        (exact for lr=1, approximation for lr>1)
        """
        return (
            self.get_current_inventory()
            + self.past_regular_orders.sum(dim=1, keepdim=True)
        )

    @abstractmethod
    def order(self):
        pass


class CyclicSlowFastModel(BaseSlowFastModel):
    """
    Cyclic slow–fast sourcing environment.

    - Slow (regular) supplier:
        * lead time = slow_lead_time
        * orders allowed only every `cycle` periods
    - Fast (expedited) supplier:
        * lead time = fast_lead_time
        * orders allowed every period
    """

    def __init__(
        self,
        cycle: int,
        slow_lead_time: int,
        fast_lead_time: int,
        slow_order_cost: float,
        fast_order_cost: float,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator,
        batch_size: int = 1,
    ):
        self.cycle = cycle
        self.t = 0  # time index

        super().__init__(
            slow_lead_time=slow_lead_time,
            fast_lead_time=fast_lead_time,
            slow_order_cost=slow_order_cost,
            fast_order_cost=fast_order_cost,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            init_inventory=init_inventory,
            demand_generator=demand_generator,
            batch_size=batch_size,
        )

    # ---------- cycle helpers ----------

    def reset(self, batch_size: int | None = None) -> None:
        super().reset(batch_size)
        self.t = 0

    def get_cycle_phase(self) -> int:
        """
        Returns current cycle phase: t mod cycle
        """
        return self.t % self.cycle

    def slow_order_allowed(self) -> bool:
        """
        Slow orders are allowed only at the beginning of each cycle.
        """
        return self.get_cycle_phase() == 0

    # ---------- core dynamics ----------

    def order(
        self,
        slow_q: torch.Tensor,
        fast_q: torch.Tensor,
        seed: int | None = None,
    ) -> None:
        """
        Place slow and fast orders, enforce cyclic constraint,
        update inventory and advance time by one period.
        """

        if seed is not None:
            torch.manual_seed(seed)

        # ensure tensor shape
        if not isinstance(slow_q, torch.Tensor):
            slow_q = torch.tensor([[slow_q]])
        if not isinstance(fast_q, torch.Tensor):
            fast_q = torch.tensor([[fast_q]])

        # enforce cyclic constraint on slow orders
        if not self.slow_order_allowed():
            slow_q = torch.zeros_like(slow_q)

        # record orders
        self.past_regular_orders = torch.cat(
            [self.past_regular_orders, slow_q], dim=1
        )
        self.past_expedited_orders = torch.cat(
            [self.past_expedited_orders, fast_q], dim=1
        )

        # arrivals from slow pipeline
        if self.past_regular_orders.shape[1] >= 1 + self.get_slow_lead_time():
            arrived_slow = self.past_regular_orders[
                :, [-1 - self.get_slow_lead_time()]
            ]
        else:
            arrived_slow = torch.zeros(self.batch_size, 1)

        # arrivals from fast pipeline
        if self.past_expedited_orders.shape[1] >= 1 + self.get_fast_lead_time():
            arrived_fast = self.past_expedited_orders[
                :, [-1 - self.get_fast_lead_time()]
            ]
        else:
            arrived_fast = torch.zeros(self.batch_size, 1)

        # demand realization
        demand = self.demand_generator.sample(self.batch_size)
        self.past_demands = torch.cat([self.past_demands, demand], dim=1)

        # inventory update
        current_inventory = (
            self.get_current_inventory()
            + arrived_slow
            + arrived_fast
            - demand
        )

        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )

        # advance time
        self.t += 1

