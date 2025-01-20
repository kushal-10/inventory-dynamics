import torch

from ..sourcing_model import SingleSourcingModel
from .base import BaseSingleController


class BaseStockController(BaseSingleController):
    """
    Base stock controller for single-sourcing inventory optimization.
    """

    def __init__(self):
        self.sourcing_model = None
        self.z_star = None

    def fit(
        self,
        sourcing_model: SingleSourcingModel,
        num_samples: int = 100000,
        seed: int = None,
    ):
        """
        Calculate the optimal target inventory level z* and store it in self.z_star.

        Returns
        -------
        None
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.sourcing_model = sourcing_model

        # Generate samples for l + 1 periods
        samples = sourcing_model.demand_generator.sample(
            batch_size=num_samples, batch_width=self.sourcing_model.get_lead_time() + 1
        )

        # Calculate the total demand for each sample
        total_demand_samples = samples.sum(dim=1)

        # Get shortage cost and holding cost from sourcing model
        b = sourcing_model.get_shortage_cost()
        h = sourcing_model.get_holding_cost()

        # Calculate z* using the empirical percentile (inverse CDF)
        service_level = b / (b + h)
        self.z_star = (
            torch.quantile(total_demand_samples.float(), service_level).int().item()
        )

    def predict(self, current_inventory, past_orders=None, output_tensor=False):
        """
        Calculate the replenishment order quantity.

        Parameters
        ----------
        current_inventory : int
            Current inventory level.
        past_orders : list, or torch.Tensor, optional
            Array of past orders. If `past_orders` is None, or the length of `past_orders` is lower than `lead_time`, it will be padded with zeros. If the length of `past_orders` is higher than `lead_time`, only the last `lead_time` orders will be used during inference.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        Returns
        -------
        float
            The replenishment order quantity.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        lead_time = self.sourcing_model.get_lead_time()

        current_inventory = self._current_inventory_check(current_inventory)
        past_orders = self._past_orders_check(past_orders, lead_time)

        if lead_time == 0:
            inventory_position = current_inventory
        elif lead_time > 0:
            inventory_position = current_inventory + past_orders[:, -lead_time:].sum(
                dim=1, keepdim=True
            )
        else:
            raise ValueError("`lead_time` cannot be less than 0")

        result = torch.relu(self.z_star - inventory_position)

        if output_tensor:
            return result
        else:
            return result.int().item()

    def reset(self):
        self.z_star = None
        self.sourcing_model = None
