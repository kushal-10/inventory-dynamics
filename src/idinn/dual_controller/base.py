from abc import ABCMeta, abstractmethod
import torch


class BaseDualController(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, sourcing_model):
        """
        Fit the controller to the sourcing model.
        """
        pass

    @abstractmethod
    def predict(self, current_inventory, past_regular_orders=None, past_expedited_orders=None, output_tensor=False):
        """
        Predict the replenishment order quantity.

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
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the controller to the initial state.
        """
        pass

    def _check_current_inventory(self, current_inventory):
        """
        Check and convert types of `current_inventory` for `predict()`.
        """
        if isinstance(current_inventory, int):
            current_inventory = torch.tensor([[current_inventory]], dtype=torch.float32)
        elif isinstance(current_inventory, torch.Tensor):
            pass
        else:
            raise TypeError("`current_inventory`'s type is not supported.")

        return current_inventory
    
    def _check_past_orders(self, past_orders, lead_time):
        """
        Check and convert types of `past_regular_orders` for `predict()`.
        Pad `past_orders` with zeros if either is too short.
        """
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
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )
        return last_cost

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
                current_inventory, past_regular_orders, past_expedited_orders, output_tensor=True
            )
            sourcing_model.order(regular_q, expedited_q)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()
        return total_cost

    def get_average_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the average cost for dual-sourcing controllers.

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
        float
            The average cost.
        """
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )

    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        """
        Simulate the sourcing model's output using the given controller.

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
            regular_q, expedited_q = self.predict(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
        past_inventories = sourcing_model.get_past_inventories()[0, :].detach().cpu().numpy()
        past_regular_orders = (
            sourcing_model.get_past_regular_orders()[0, :].detach().cpu().numpy()
        )
        past_expedited_orders = (
            sourcing_model.get_past_expedited_orders()[0, :].detach().cpu().numpy()
        )
        return past_inventories, past_regular_orders, past_expedited_orders

    def plot(self, sourcing_model, sourcing_periods, linewidth=1, seed=None):
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
        seed : int, optional
            Random seed for reproducibility.
        """
        from matplotlib import pyplot as plt

        past_inventories, past_regular_orders, past_expedited_orders = self.simulate(
            sourcing_model=sourcing_model, sourcing_periods=sourcing_periods, seed=seed
        )
        fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].step(
            range(sourcing_periods),
            past_inventories[-sourcing_periods:],
            linewidth=linewidth,
            color="tab:blue",
        )
        ax[0].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_title("Inventory")
        ax[0].set_xlabel("Period")
        ax[0].set_ylabel("Quantity")

        ax[1].step(
            range(sourcing_periods),
            past_expedited_orders[-sourcing_periods:],
            label="Expedited Order",
            linewidth=linewidth,
            color="tab:green",
        )
        ax[1].step(
            range(sourcing_periods),
            past_regular_orders[-sourcing_periods:],
            label="Regular Order",
            linewidth=linewidth,
            color="tab:orange",
        )
        ax[1].yaxis.get_major_locator().set_params(integer=True)
        ax[1].set_title("Order")
        ax[1].set_xlabel("Period")
        ax[1].set_ylabel("Quantity")
        ax[1].legend()

    def save(self, path):
        """
        Save the controller to the specified path.
        """
        import pickle
        
        pickle.dump(self, path)

    def load(self, path):
        """
        Load the controller from the specified path.
        """
        import pickle

        pickle.load(path)