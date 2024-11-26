import numpy as np
import torch
from .base import BaseSingleController


class SingleSourcingNeuralController(torch.nn.Module, BaseSingleController):
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
        self.sourcing_model = None
        self.nn = None

    def init_layers(self):
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
        lead_time = self.sourcing_model.get_lead_time()
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
        self.nn = torch.nn.Sequential(*architecture)

    def predict(
        self,
        current_inventory,
        past_orders=None,
    ):
        """
        Perform forward pass through the neural network.

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory levels.
        past_orders : list, or torch.Tensor, optional
            Past order quantities.

        Returns
        -------
        torch.Tensor
            Order quanty calculated by the neural network.
        """
        if self.sourcing_model is None:
            raise ValueError("Sourcing model is not availble.")

        if self.nn is None:
            self.init_layers()
        
        if not isinstance(current_inventory, torch.Tensor):
            current_inventory = torch.tensor([[current_inventory]], dtype=torch.float32)
        if not isinstance(past_orders, torch.Tensor):
            past_orders = torch.tensor([past_orders], dtype=torch.float32)

        # Get lead time from self.sourcing model
        lead_time = self.sourcing_model.get_lead_time()
    
        if lead_time > 0:
            inputs = torch.cat(
                [current_inventory, past_orders[:, -lead_time :]], dim=1
            )
        else:
            inputs = current_inventory
        h = self.nn(inputs)
        q = h - torch.frac(h).clone().detach()
        return q

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
        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.nn is None:
            self.init_layers()

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

    def reset(self):
        self.sourcing_model = None
        self.nn = None
