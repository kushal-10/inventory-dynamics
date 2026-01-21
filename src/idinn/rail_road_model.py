from typing import Optional, Union

import torch

from .demand import CustomDemand, UniformDemand
from .demand_three_subperiods import DiscreteTruncatedGammaDemand


class RailRoadInventoryModel:
    """
    Periodic rail-road inventory model.

    - Time is discrete (daily)
    - Demand occurs every day
    - Orders arrive immediately (lead time = 0)
    - Regular (rail) orders are allowed only every K days
    - Expedited (road) orders are allowed on all other days
    """

    def __init__(
        self,
        cycle_length: int,
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
        """
        Parameters
        ----------
        cycle_length : int
            Length of the ordering cycle (e.g. 3 means rail every 3 days).
        regular_order_cost : float
            Unit cost for regular (rail) orders.
        expedited_order_cost : float
            Unit cost for expedited (road) orders.
        holding_cost : float
            Per-unit holding cost.
        shortage_cost : float
            Per-unit shortage (backlog) cost.
        init_inventory : float
            Initial inventory level.
        demand_generator : DemandGenerator
            Demand generator instance.
        batch_size : int, default=1
            Batch size for simulation / control.
        """
        if cycle_length <= 0:
            raise ValueError("`cycle_length` must be a positive integer.")

        self.cycle_length = cycle_length
        self.regular_order_cost = regular_order_cost
        self.expedited_order_cost = expedited_order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.demand_generator = demand_generator
        self.batch_size = batch_size

        self.init_inventory = torch.tensor(
            [init_inventory], dtype=torch.float, requires_grad=True
        )

        self.reset()

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None and batch_size != self.batch_size:
            self.batch_size = batch_size

        # Inventory history
        self.past_inventories = self.get_init_inventory().repeat(self.batch_size, 1)

        # Demand history
        self.past_demands = torch.zeros(self.batch_size, 1)

        # Order history (single stream, since only one mode per day)
        self.past_orders = torch.zeros(self.batch_size, 1)

        # Day-in-cycle (τ_t)
        self.current_cycle_day = 0

    def get_init_inventory(self) -> torch.Tensor:
        # Ensure integer-valued initial inventory
        return self.init_inventory - torch.frac(self.init_inventory).detach()

    def get_current_inventory(self) -> torch.Tensor:
        return self.past_inventories[:, [-1]]

    def get_current_cycle_day(self) -> int:
        return self.current_cycle_day

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
        """
        Returns True if regular (rail) ordering is allowed today.
        """
        return self.current_cycle_day == 0

    def order(self, q: torch.Tensor, seed: Optional[int] = None) -> None:
        """
        Place an order and update inventory.

        Parameters
        ----------
        q : torch.Tensor
            Order quantity (non-negative).
        seed : int, optional
            Random seed for demand generation.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor([[q]])

        # Determine ordering cost (handled externally by controller)
        self.past_orders = torch.cat([self.past_orders, q], dim=1)

        # Generate demand
        current_demand = self.demand_generator.sample(self.batch_size)

        self.past_demands = torch.cat([self.past_demands, current_demand], dim=1)

        # Inventory update (lead time = 0)
        current_inventory = self.get_current_inventory() + q - current_demand

        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )

        # Advance cycle day
        self.current_cycle_day = (self.current_cycle_day + 1) % self.cycle_length

class RailRoadInventoryModelWithLeadTime:
    """
    Periodic rail-road inventory model with lead time > 0.

    - Time is discrete
    - Demand occurs every period
    - Orders arrive after L periods
    - Rail orders allowed only on cycle day 0
    - Road orders allowed on all other days
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
        demand_generator,
        batch_size: int = 1,
    ):
        if cycle_length <= 0:
            raise ValueError("`cycle_length` must be positive.")
        if lead_time <= 0:
            raise ValueError("`lead_time` must be positive.")

        self.cycle_length = cycle_length
        self.lead_time = lead_time

        self.regular_order_cost = regular_order_cost
        self.expedited_order_cost = expedited_order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

        self.demand_generator = demand_generator
        self.batch_size = batch_size

        self.init_inventory = torch.tensor(
            [init_inventory], dtype=torch.float, requires_grad=True
        )

        self.reset()

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self.batch_size = batch_size

        self.past_inventories = self.get_init_inventory().repeat(self.batch_size, 1)
        self.past_demands = torch.zeros(self.batch_size, 1)

        # Pipeline of future arrivals
        self.past_orders = torch.zeros(self.batch_size, self.lead_time)

        # Cycle day τ_t
        self.current_cycle_day = 0

    def get_init_inventory(self) -> torch.Tensor:
        return self.init_inventory - torch.frac(self.init_inventory).detach()

    def get_current_inventory(self) -> torch.Tensor:
        return self.past_inventories[:, [-1]]

    def get_current_cycle_day(self) -> int:
        return self.current_cycle_day

    def get_pipeline(self) -> torch.Tensor:
        """
        Orders that will arrive in the future (length = lead_time).
        """
        return self.past_orders

    # ------------------------------------------------------------------
    # Costs
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
        Place an order and update inventory with lead time.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor([[q]])

        # Orders enter the pipeline
        self.past_orders = torch.cat(
            [self.past_orders[:, 1:], q], dim=1
        )

        # Arriving order
        arrived = self.past_orders[:, [0]]

        # Demand
        demand = self.demand_generator.sample(self.batch_size)
        self.past_demands = torch.cat([self.past_demands, demand], dim=1)

        # Inventory update
        current_inventory = (
            self.get_current_inventory() + arrived - demand
        )
        self.past_inventories = torch.cat(
            [self.past_inventories, current_inventory], dim=1
        )

        # Advance cycle
        self.current_cycle_day = (self.current_cycle_day + 1) % self.cycle_length
