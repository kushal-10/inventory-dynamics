import matplotlib.pyplot as plt
import numpy as np
import torch


class NeuralControllerMixIn:
    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))


class SingleSourcingNeuralController(torch.nn.Module, NeuralControllerMixIn):
    """
    SingleSourcingNeuralController is a neural network-based controller for inventory optimization in a single-sourcing scenario.

    Parameters
    ----------
    hidden_layers : list, default is [2]
        List of integers representing the number of units in each hidden layer.
    activation : torch.nn.Module, default is torch.nn.CELU(alpha=1)
        Activation function to be used in the hidden layers.

    Attributes
    ----------
    hidden_layers : list
        List of integers representing the number of units in each hidden layer.
    activation : torch.nn.Module
        Activation function used in the hidden layers.
    architecture : torch.nn.Sequential
        Sequential stack of linear layers and activation functions.

    Methods
    -------
    init_layers(lead_time)
        Initialize the layers of the neural network based on the lead time.
    forward(current_inventory, past_orders)
        Perform forward pass through the neural network.
    get_total_cost(sourcing_model, sourcing_periods, seed=None)
        Calculate the total cost over a given number of sourcing periods.
    train(sourcing_model, sourcing_periods, epochs, ...)
        Train the neural network controller using the sourcing model and specified parameters.
    simulate(sourcing_model, sourcing_periods)
        Simulate the inventory and order quantities over a given number of sourcing periods.
    plot(sourcing_model, sourcing_periods)
        Plot the inventory and order quantities over a given number of sourcing periods.
    """

    def __init__(self, hidden_layers=[2], activation=torch.nn.CELU(alpha=1)):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.lead_time = None
        self.architecture = None

    def init_layers(self, lead_time):
        """
        Initialize the layers of the neural network based on the lead time.

        Parameters
        ----------
        lead_time : int
            The lead time for sourcing.

        Returns
        -------
        None
        """
        self.lead_time = lead_time
        architecture = [
            torch.nn.Linear(lead_time + 1, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], 1, bias=False),
            torch.nn.ReLU(),
        ]
        self.architecture = torch.nn.Sequential(*architecture)

    def forward(
        self,
        current_inventory,
        past_orders,
    ):
        """
        Perform forward pass through the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory levels.
        past_orders : int, or torch.Tensor
            Past order quantities.

        Returns
        -------
        torch.Tensor
            Order quanty calculated by the neural network.
        """
        if not isinstance(current_inventory, torch.Tensor):
            current_inventory = torch.tensor([[current_inventory]], dtype=torch.float32)
        if not isinstance(past_orders, torch.Tensor):
            past_orders = torch.tensor([past_orders], dtype=torch.float32)
        if self.lead_time > 0:
            inputs = torch.cat(
                [current_inventory, past_orders[:, -self.lead_time :]], dim=1
            )
        else:
            inputs = current_inventory
        h = self.architecture(inputs)
        q = h - torch.frac(h).clone().detach()
        return q

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the total cost over a given number of sourcing periods.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model to be used for cost calculation.
        sourcing_periods : int
            The number of sourcing periods.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        numpy.ndarray
            Total cost over the sourcing periods.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.architecture is None:
            self.init_layers(sourcing_model.get_lead_time())

        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.forward(current_inventory, past_orders)
            sourcing_model.order(q)
            current_cost = sourcing_model.get_cost()
            total_cost += current_cost.mean()
        return total_cost

    def fit(
        self,
        sourcing_model,
        sourcing_periods,
        epochs,
        validation_sourcing_periods=None,
        validation_freq=10,
        init_inventory_lr=1e-1,
        init_inventory_freq=4,
        parameters_lr=3e-3,
        tensorboard_writer=None,
        seed=None,
    ):
        """
        Train the neural network controller using the sourcing model and specified parameters.

        Parameters
        ----------
        sourcing_model : SourcingModel
            The sourcing model for training.
        sourcing_periods : int
            The number of sourcing periods for training.
        epochs : int
            The number of training epochs.
        validation_sourcing_periods : int, optional
            The number of sourcing periods for validation.
        validation_freq : int, default is 10
            Only relevant if `validation_sourcing_periods` is provided. Specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=10` runs validation every 10 epochs.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory is updated. e.g. `init_inventory_freq=4` updates initial inventory after updating parameters for 4 epochs.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 3e-3
            Learning rate for updating neural network parameters.
        tensorboard_writer : tensorboard.SummaryWriter, optional
            Tensorboard writer for logging.
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.architecture is None:
            self.init_layers(sourcing_model.get_lead_time())

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
        min_cost = np.inf

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_parameters.zero_grad()
            optimizer_init_inventory.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
            total_cost.backward()
            # Gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % 10 == 0:
                eval_cost = self.get_total_cost(
                    sourcing_model, validation_sourcing_periods
                )
                if eval_cost < min_cost:
                    min_cost = eval_cost
                    best_state = self.state_dict()
            else:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", total_cost / sourcing_periods, epoch
                )
                if (
                    validation_sourcing_periods is not None
                    and epoch % validation_freq == 0
                ):
                    # Log validation loss
                    eval_cost = self.get_total_cost(
                        sourcing_model, validation_sourcing_periods
                    )
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_cost / validation_sourcing_periods,
                        epoch,
                    )
                tensorboard_writer.flush()
        # Load the best model
        self.load_state_dict(best_state)

    def simulate(self, sourcing_model, sourcing_periods, seed=None):
        """
        Simulate the inventory and order quantities over a given number of sourcing periods.

        Parameters
        ----------
        sourcing_model : SingleSourcingModel
            The sourcing model to be used for simulation.
        sourcing_periods : int
            The number of sourcing periods for simulation.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple
            A tuple containing the past inventories and past orders as numpy arrays.
        """
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_orders = sourcing_model.get_past_orders()
            q = self.forward(current_inventory, past_orders)
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


class DualSourcingNeuralController(torch.nn.Module, NeuralControllerMixIn):
    """
    DualSourcingNeuralController is a neural network controller for dual sourcing inventory optimization.

    Parameters
    ----------
    hidden_layers : list, default is [128, 64, 32, 16, 8, 4]
        List of integers specifying the sizes of hidden layers.
    activation : torch.nn.Module, default is torch.nn.CELU(alpha=1)
        Activation function to be used in the hidden layers.
    compressed : bool, default is False
        Flag indicating whether the input is compressed.

    Attributes
    ----------
    hidden_layers : list
        List of integers specifying the sizes of hidden layers.
    activation : torch.nn.Module
        Activation function to be used in the hidden layers.
    compressed : bool
        Flag indicating whether the input is compressed.
    regular_lead_time : int
        Regular lead time.
    expedited_lead_time : int
        Expedited lead time.
    architecture : torch.nn.Sequential
        Sequential stack of linear layers and activation functions.

    Methods
    -------
    init_layers(regular_lead_time, expedited_lead_time)
        Initialize the layers of the neural network.
    forward(current_inventory, past_orders)
        Forward pass of the neural network.
    get_total_cost(sourcing_model, sourcing_periods, seed=None)
        Calculate the total cost of the sourcing model.
    train(sourcing_model, sourcing_periods, epochs, ...)
        Trains the neural network controller using the sourcing model and specified parameters.
    simulate(sourcing_model, sourcing_periods, seed=None)
        Simulate the sourcing model using the neural network.
    plot(sourcing_model, sourcing_periods)
        Plot the inventory and order quantities.
    """

    def __init__(
        self,
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.ReLU(),
        compressed=False,
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed
        self.lead_time = None
        self.architecture = None

    def init_layers(self, regular_lead_time, expedited_lead_time):
        """
        Initialize the layers of the neural network.

        Parameters
        ----------
        regular_lead_time : int
            Regular lead time.
        expedited_lead_time : int
            Expedited lead time.
        """
        self.regular_lead_time = regular_lead_time
        self.expedited_lead_time = expedited_lead_time
        if self.compressed:
            input_length = regular_lead_time + expedited_lead_time
        else:
            input_length = regular_lead_time + expedited_lead_time + 1

        architecture = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers)):
            if i < len(self.hidden_layers) - 1:
                architecture += [
                    torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                    self.activation,
                ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], 2),
            torch.nn.ReLU(),
        ]
        self.architecture = torch.nn.Sequential(*architecture)

    def forward(self, current_inventory, past_regular_orders, past_expedited_orders):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : int, or torch.Tensor
            Past regular orders.
        past_expedited_orders : int, or torch.Tensor
            Past expedited orders.

        Returns
        -------
        regular_q : torch.Tensor
            Regular order quantity.
        expedited_q : torch.Tensor
            Expedited order quantity.
        """
        if not isinstance(current_inventory, torch.Tensor):
            current_inventory = torch.tensor([[current_inventory]], dtype=torch.float32)
        if not isinstance(past_regular_orders, torch.Tensor):
            past_regular_orders = torch.tensor(
                [past_regular_orders], dtype=torch.float32
            )
        if not isinstance(past_expedited_orders, torch.Tensor):
            past_expedited_orders = torch.tensor(
                [past_expedited_orders], dtype=torch.float32
            )

        if self.regular_lead_time > 0:
            if self.compressed:
                inputs = past_regular_orders[:, -self.regular_lead_time :]
                inputs[:, 0] += current_inventory
            else:
                inputs = torch.cat(
                    [
                        current_inventory,
                        past_regular_orders[:, -self.regular_lead_time :],
                    ],
                    dim=1,
                )
        else:
            inputs = current_inventory

        if self.expedited_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_expedited_orders[:, -self.expedited_lead_time :]], dim=1
            )

        h = self.architecture(inputs)
        q = h - torch.frac(h).clone().detach()
        regular_q = q[:, [0]]
        expedited_q = q[:, [1]]
        return regular_q, expedited_q

    def get_total_cost(self, sourcing_model, sourcing_periods, seed=None):
        """
        Calculate the total cost of the sourcing model.

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
        total_cost : torch.Tensor
            Total cost.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.architecture is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        total_cost = 0
        for i in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.forward(
                current_inventory, past_regular_orders, past_expedited_orders
            )
            sourcing_model.order(regular_q, expedited_q)
            current_cost = sourcing_model.get_cost(regular_q, expedited_q)
            total_cost += current_cost.mean()
        return total_cost

    def fit(
        self,
        sourcing_model,
        sourcing_periods,
        epochs,
        validation_sourcing_periods=None,
        validation_freq=50,
        init_inventory_freq=4,
        init_inventory_lr=1e-1,
        parameters_lr=3e-3,
        tensorboard_writer=None,
        seed=None,
    ):
        """
        Train the neural network controller using the sourcing model and specified parameters.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model for training.
        sourcing_periods : int
            Number of sourcing periods for training.
        epochs : int
            Number of training epochs.
        validation_sourcing_periods : int, optional
            Number of sourcing periods for validation.
        validation_freq : int, default is 10
            Only relevant if `validation_sourcing_periods` is provided. Specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=10` runs validation every 10 epochs.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory is updated. e.g. `init_inventory_freq=4` updates initial inventory after updating parameters for 4 epochs.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 3e-3
            Learning rate for updating neural network parameters.
        tensorboard_writer : tensorboard.SummaryWriter, optional
        seed : int, optional
            Random seed for reproducibility.
            Tensorboard writer for logging.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self.architecture is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
        min_cost = np.inf

        for epoch in range(epochs):
            # Clear grad cache
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            total_cost = self.get_total_cost(sourcing_model, sourcing_periods)
            total_cost.backward()
            # Perform gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_cost = self.get_total_cost(
                    sourcing_model, validation_sourcing_periods
                )
                if eval_cost < min_cost:
                    min_cost = eval_cost
                    best_state = self.state_dict()
            else:
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", total_cost / sourcing_periods, epoch
                )
                if validation_sourcing_periods is not None and epoch % 10 == 0:
                    # Log validation loss
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_cost / validation_sourcing_periods,
                        epoch,
                    )
                tensorboard_writer.flush()

        self.load_state_dict(best_state)

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
        import matplotlib as mpl

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
