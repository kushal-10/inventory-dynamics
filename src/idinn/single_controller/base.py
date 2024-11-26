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

    def get_last_cost(self, sourcing_model):
        """
        Calculate the cost for the latest period of the sourcing model.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model.

        Returns
        -------
        float
            The last cost.

        """
        shortage_cost = sourcing_model.get_shortage_cost()
        holding_cost = sourcing_model.get_holding_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = holding_cost * torch.relu(
            current_inventory
        ) + shortage_cost * torch.relu(-current_inventory)
        return last_cost
    
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
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
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
    
    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        """
        Simulate the sourcing model's output using the given controller.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        past_inventories : list
            List of past inventories.
        past_orders : list
            List of past orders.

        """
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.predict(current_inventory, past_orders)
            sourcing_model.order(q)
        past_inventories = sourcing_model.get_past_inventories()[0, :].detach().numpy()
        past_orders = sourcing_model.get_past_orders()[0, :].detach().numpy()
        return past_inventories, past_orders

    def plot(self, sourcing_model, sourcing_periods, linewidth=1):
        """
        Plot the inventory and order quantities over a given number of sourcing periods.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model to be used for plotting.
        sourcing_periods : int
            The number of sourcing periods for plotting.
        linewidth : int, default is 1
            The width of the line in the step plots.
        """
        from matplotlib import pyplot as plt

        past_inventories, past_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

        ax[0].step(range(sourcing_periods), past_inventories[-sourcing_periods:], linewidth=linewidth, color='tab:blue')
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(range(sourcing_periods), past_orders[-sourcing_periods:], linewidth=linewidth, color='tab:orange')
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")