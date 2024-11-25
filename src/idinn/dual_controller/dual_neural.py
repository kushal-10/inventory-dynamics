import numpy as np
import torch
from .base import BaseDualController


class DualSourcingNeuralController(torch.nn.Module, BaseDualController):
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
            # TODO: Mention this ReLU layer in documentation
            torch.nn.ReLU(),
        ]
        self.architecture = torch.nn.Sequential(*architecture)

    def predict(self, current_inventory, past_regular_orders, past_expedited_orders):
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
    
    def reset(self):
        #TODO: Add reset function
        pass
    
    def save(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))