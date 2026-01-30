from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ..slow_fast import CyclicSlowFastModel


class BaseSlowFastController(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, sourcing_model: CyclicSlowFastModel, **kwargs) -> None:
        """
        Fit the controller to the slow–fast sourcing model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_slow_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_fast_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Predict slow and fast replenishment quantities.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        pass

    # ------------------------------------------------------------------
    # Shared utilities (mirrors BaseDualController, renamed semantically)
    # ------------------------------------------------------------------

    def _check_current_inventory(
        self, current_inventory: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(current_inventory, int):
            return torch.tensor([[current_inventory]], dtype=torch.float32)
        elif isinstance(current_inventory, torch.Tensor):
            return current_inventory
        raise TypeError("`current_inventory`'s type is not supported.")

    def _check_past_orders(
        self, past_orders: Optional[Union[List[int], torch.Tensor]], lead_time: int
    ) -> torch.Tensor:
        if past_orders is None:
            past_orders = torch.zeros(1, lead_time)
        elif isinstance(past_orders, list):
            past_orders = torch.tensor([past_orders], dtype=torch.float32)
        elif isinstance(past_orders, torch.Tensor):
            pass
        else:
            raise TypeError("`past_orders`'s type is not supported.")

        order_len = past_orders.shape[1]
        if order_len < lead_time:
            return torch.nn.functional.pad(past_orders, (lead_time - order_len, 0))
        else:
            return past_orders

    def get_last_cost(self, sourcing_model: CyclicSlowFastModel) -> torch.Tensor:
        """
        Cost incurred in the most recent period.
        """
        last_slow_q = sourcing_model.get_last_slow_order()
        last_fast_q = sourcing_model.get_last_fast_order()

        slow_cost = sourcing_model.get_slow_order_cost()
        fast_cost = sourcing_model.get_fast_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()

        current_inventory = sourcing_model.get_current_inventory()

        return (
            slow_cost * last_slow_q
            + fast_cost * last_fast_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_slow_orders = sourcing_model.get_slow_pipeline()
            past_fast_orders = sourcing_model.get_fast_pipeline()

            slow_q, fast_q = self.predict(
                current_inventory,
                past_slow_orders,
                past_fast_orders,
                output_tensor=True,
            )

            sourcing_model.order(slow_q, fast_q)
            total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return self.get_total_cost(sourcing_model, sourcing_periods, seed) / sourcing_periods

    @no_type_check
    def simulate(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:

        if seed is not None:
            torch.manual_seed(seed)

        sourcing_model.reset(batch_size=1)

        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_slow_orders = sourcing_model.get_slow_pipeline()
            past_fast_orders = sourcing_model.get_fast_pipeline()

            slow_q, fast_q = self.predict(
                current_inventory, past_slow_orders, past_fast_orders
            )
            sourcing_model.order(slow_q, fast_q)

        return (
            sourcing_model.get_past_inventories()[0].detach().cpu().numpy(),
            sourcing_model.get_slow_pipeline()[0].detach().cpu().numpy(),
            sourcing_model.get_fast_pipeline()[0].detach().cpu().numpy(),
        )

    def plot(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        linewidth: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:

        inv, slow, fast = self.simulate(
            sourcing_model, sourcing_periods, seed
        )

        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

        ax[0].step(range(len(inv)), inv[-sourcing_periods:], linewidth=linewidth)
        ax[0].set_title("Inventory")

        ax[1].step(range(len(slow)), slow[-sourcing_periods:], label="Slow")
        ax[1].step(range(len(fast)), fast[-sourcing_periods:], label="Fast")
        ax[1].legend()
        ax[1].set_title("Orders")

        return fig, ax
