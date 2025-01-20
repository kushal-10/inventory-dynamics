from typing import List, Optional, Tuple, Union

import torch

from ..sourcing_model import DualSourcingModel
from .base import BaseDualController


class CappedDualIndexController(BaseDualController):
    """
    Controller class for capped dual index inventory optimization.

    Parameters
    ----------
    s_e : int
        Capped dual index parameter 1
    s_r : int
        Capped dual index parameter 2
    q_r : int
        Capped dual index parameter 3

    Notes
    -----
    The function follows the implementation of Sun, J., & Van Mieghem, J. A. (2019)([1]_).

    References
    ----------
    .. [1] Robust dual sourcing inventory management: Optimality of capped dual index policies and smoothing.
           Manufacturing & Service Operations Management, 21(4), 912-931.
    """

    def __init__(self, s_e: int = 0, s_r: int = 0, q_r: int = 0) -> None:
        self.s_e = s_e
        self.s_r = s_r
        self.q_r = q_r
        self.sourcing_model: Optional[DualSourcingModel] = None

    def capped_dual_index_sum(
        self,
        current_inventory: int,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        limit: bool = False,
    ) -> int:
        """
        Calculate the capped dual index sum.

        Parameters
        ----------
        current_inventory : int
            Current inventory level.
        past_regular_orders : torch.Tensor
            Array of past regular orders.
        past_expedited_orders : torch.Tensor
            Array of past expedited orders.
        regular_lead_time : int
            Regular lead time.
        expedited_lead_time : int
            Expedited lead time.
        limit : bool
            If true, set parameter k for capped dual index sum calculation, where 0 <= k <= l_r -1,
            to regular_lead_time - expedited_lead_time - 1. Else, set k to 0.

        Returns
        -------
        int
            The capped dual index sum.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        regular_lead_time = self.sourcing_model.get_regular_lead_time()
        expedited_lead_time = self.sourcing_model.get_expedited_lead_time()

        current_inventory = self._current_inventory_check(current_inventory)
        past_regular_orders = self._past_orders_check(
            past_regular_orders, regular_lead_time
        )
        past_expedited_orders = self._past_orders_check(
            past_expedited_orders, expedited_lead_time
        )

        if limit:
            k = regular_lead_time - expedited_lead_time - 1
        else:
            k = 0

        inventory_position = (
            current_inventory
            + past_regular_orders[
                :, -regular_lead_time : -regular_lead_time + k + 1
            ].sum()
        )

        if expedited_lead_time >= 1:
            inventory_position += past_expedited_orders[
                -expedited_lead_time : +min(k - expedited_lead_time, -1) + 1
            ].sum()

        return inventory_position

    def fit(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        s_e_range: torch.Tensor = torch.arange(2, 11),
        s_r_range: torch.Tensor = torch.arange(2, 11),
        q_r_range: torch.Tensor = torch.arange(2, 11),
        seed: Optional[int] = None,
    ) -> None:
        """
        Train the capped dual index controller.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        s_e_range : torch.Tensor, optional
            Range of values for s_e.
        s_r_range : torch.Tensor, optional
            Range of values for s_r.
        q_r_range : torch.Tensor, optional
            Range of values for q_r.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        min_cost = torch.inf
        for s_e in s_e_range:
            for s_r in s_r_range:
                for q_r in q_r_range:
                    sourcing_model.reset()
                    self.s_e = s_e
                    self.s_r = s_r
                    self.q_r = q_r
                    total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
                    if total_cost < min_cost:
                        min_cost = total_cost
                        s_e_optimal = s_e
                        s_r_optimal = s_r
                        q_r_optimal = q_r
        self.s_e = s_e_optimal
        self.s_r = s_r_optimal
        self.q_r = q_r_optimal

    def predict(
        self,
        current_inventory: int,
        past_regular_orders: Optional[torch.Tensor] = None,
        past_expedited_orders: Optional[torch.Tensor] = None,
        output_tensor: bool = False
    ) -> Tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]:
        """
        Perform forward calculation for capped dual index optimization.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. If the length of `past_regular_orders` is lower than `regular_lead_time`, it will be padded with zeros. If the length of `past_regular_orders` is higher than `regular_lead_time`, only the last `regular_lead_time` orders will be used during inference.
        past_expedited_orders : list, or torch.Tensor, optional
            Past expedited orders. If the length of `past_expedited_orders` is lower than `expedited_lead_time`, it will be padded with zeros. If the length of `past_expedited_orders` is higher than `expedited_lead_time`, only the last `expedited_lead_time` orders will be used during inference.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        Returns
        -------
        tuple
            A tuple containing the regular order quantity and expedited order quantity.
        """
        inventory_position = self.capped_dual_index_sum(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            limit=False,
        )
        inventory_position_lm1 = self.capped_dual_index_sum(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            limit=True,
        )
        regular_q = int(min(max(0, self.s_r - inventory_position_lm1), self.q_r))
        expedited_q = int(max(0, self.s_e - inventory_position))

        if output_tensor:
            return torch.tensor([[regular_q]]), torch.tensor([[expedited_q]])
        else:
            return regular_q, expedited_q

    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        self.s_e = 0
        self.s_r = 0
        self.q_r = 0
        self.sourcing_model = None
