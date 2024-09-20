from abc import ABCMeta, abstractmethod
import torch


class BaseSingleController(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, sourcing_model, num_samples=100000):
        """
        Fit the controller to the sourcing model.
        """
        pass

    @abstractmethod
    def predict(self, current_inventory, past_orders):
        """
        Predict the replenishment order quantity.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller to the initial state.
        """
        pass

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the total cost for single-sourcing optimization.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        float
            The total cost.
        """
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q_t = self.predict(current_inventory, past_orders)
            sourcing_model.order(q_t)
            current_cost = sourcing_model.get_cost()
            total_cost += current_cost
        return total_cost
    
    def get_average_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the average cost for single-sourcing optimization.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        float
            The average cost.
        """
        return self.get_total_cost(sourcing_model, sourcing_periods, seed)/sourcing_periods
