from abc import ABCMeta, abstractmethod
import torch


class BaseDualController(metaclass=ABCMeta):
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
        Calculate the total cost for dual-sourcing optimization.

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
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
            current_cost = sourcing_model.get_cost(regular_q, expedited_q)
            total_cost += current_cost.mean()
        return total_cost
    
    
    def get_average_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the average cost for Dual-sourcing optimization.

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
        Simulate the sourcing model using the neural network.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        past_inventories : list
            List of past inventories.
        past_regular_orders : list
            List of past regular orders.
        past_expedited_orders : list
            List of past expedited orders.

        """
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
        past_inventories = sourcing_model.get_past_inventories()[0, :].detach().numpy()
        past_regular_orders = (
            sourcing_model.get_past_regular_orders()[0, :].detach().numpy()
        )
        past_expedited_orders = (
            sourcing_model.get_past_expedited_orders()[0, :].detach().numpy()
        )
        return past_inventories, past_regular_orders, past_expedited_orders

    def plot(self, sourcing_model, sourcing_periods, linewidth=1):
        """
        Plot the inventory and order quantities.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model.
        sourcing_periods : int
            Number of sourcing periods.
        linewidth : int, default is 1
            Width of the line in the step plots.
        """
        from matplotlib import pyplot as plt

        past_inventories, past_regular_orders, past_expedited_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].step(range(sourcing_periods), past_inventories[-sourcing_periods:], linewidth=linewidth, color='tab:blue')
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(
            range(sourcing_periods),
            past_expedited_orders[-sourcing_periods:],
            label="Expedited Order",
            linewidth=linewidth,
            color='tab:green'
        )
        ax[1].step(
            range(sourcing_periods),
            past_regular_orders[-sourcing_periods:],
            label="Regular Order",
            linewidth=linewidth,
            color='tab:orange'
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")
        ax[1].legend()
