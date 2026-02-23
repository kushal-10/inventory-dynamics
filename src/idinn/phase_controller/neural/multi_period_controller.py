from typing import List, Optional, Tuple, Union, no_type_check
import logging 
from datetime import datetime 

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseNeuralController
from ...sourcing_model import DualSourcingModel

# Get root logger
logger = logging.getLogger()

class MultiPeriodNeuralController(torch.nn.Module, BaseNeuralController):
    """
    Implements a multi-period nerual netowrk architecture. Input consists of periodic time states
    E.g. I_t, I_(t+n), I_(t+2n)....
    Demand is realized internally for the whole cycle
    Cost calculation is based on the whole cycle
    """

    def __init__(
        self, 
        hidden_layers: List[int] = [128, 64, ],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        n_periods: int = 2
        ) -> None:

        """
        Parameters
        ----------
        hidden_layers: Architecure of hidden layers. hidden_layer[n] represents the number of nerons in layer n.
        activation: Activations betweeen layers
        n_periods: Defines the number of time periods in 1 cycle. The output heads (and accordingly, the forward pass) will enumerate based on this value.
        """
        
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation 
        self.n_periods = n_periods 
        self.MAX_Q = 20

        self.model = None

        assert self.n_periods > 1, "Periods in a cycle should be > 1"

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Build NN architecture
        """

        input_length = regular_lead_time+expedited_lead_time+1

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
            torch.nn.Linear(self.hidden_layers[-1], self.n_periods+1), # n_period outputs for Qe, +1 for Qr at t=0 in a cycle
            # TODO: Mention this ReLU layer in documentation
            torch.nn.ReLU(),
        ]

        self.model = torch.nn.Sequential(*architecture)
        
        logger.info(
            f"Initialized neural network layers with regular_lead_time={regular_lead_time}, "
            f"expedited_lead_time={expedited_lead_time}, "
            f"Periods in a Cycle : {self.n_periods}"
        )


    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        sourcing_model: DualSourcingModel,
    ) -> torch.Tensor:

        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(
            past_regular_orders, regular_lead_time
        )
        past_expedited_orders = self._check_past_orders(
            past_expedited_orders, expedited_lead_time
        )

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [
                    current_inventory,
                    past_expedited_orders[:, -expedited_lead_time:],
                ],
                dim=1,
            )
        else:
            inputs = current_inventory

        if regular_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_regular_orders[:, -regular_lead_time:]], dim=1
            )
        return inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")

        h = self.model(inputs)
        h = torch.clamp(h, 0.0, self.MAX_Q)
        q = h - torch.frac(h).clone().detach()
        regular_q = q[:, [0]]
        expedited_q = q[:, 1:] # Expedited Q0, Expedited Q1 ... depending on self.n_periods

        assert h.shape[1] == self.n_periods+1, f"Forward pass failed, check n periods and expedited q outputs. There are {self.n_periods} Periods and output tensor is of shape {h.shape}"

        return regular_q, expedited_q

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Predict order qunatities from the neural network.

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
            A tuple containing the regular order quantity, expedited order quantity at time t, and expeditied order quantity at time t+1.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model,
        )
        regular_q, expedited_q = self.forward(inputs)

        if output_tensor:
            return regular_q, expedited_q
        else:
            regular_int = int(regular_q.item())
            expedited_ints = expedited_q.squeeze(0).tolist()
            return regular_int, expedited_ints


    def fit(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 100,
        init_inventory_freq: int = 4,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 3e-3,
        seed: Optional[int] = None,
    ) -> None:
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
        log_freq : int, default is 10
            Specifies how many training epochs to run before logging the training cost.
        init_inventory_freq : int, default is 4
            Specifies how many parameter updating epochs to run before initial inventory is updated. e.g. `init_inventory_freq=4` updates initial inventory after updating parameters for 4 epochs.
        init_inventory_lr : float, default is 1e-1
            Learning rate for initial inventory.
        parameters_lr : float, default is 3e-3
            Learning rate for updating neural network parameters.
        seed : int, optional
            Random seed for reproducibility.
            Tensorboard writer for logging.
        """
        # Store sourcing model in self.sourcing_model
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.model is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        start_time = datetime.now()
        logger.info(f"Sourcing periods are reduced by a factor of {self.n_periods} to keep them aligned with other non-periodic controllers")
        logger.info(f"Starting Multi-Period dual sourcing neural network training at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"validation_periods={validation_sourcing_periods}, learning_rate={parameters_lr}"
        )

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
        min_loss = np.inf

        for epoch in tqdm(range(epochs)):
            # Clear grad cache
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset()
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods, seed=seed)
            train_loss.backward()
            # Perform gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()

            # logger.info(f"Current Loss : {train_loss}")
            # Save the best model
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                # eval_loss = self.get_total_cost(
                #     sourcing_model, validation_sourcing_periods
                # )
                eval_loss = self.get_average_cost(
                    sourcing_model, validation_sourcing_periods # Normalize with sourcing periods
                )
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = self.state_dict()
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_state = self.state_dict()
  
            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)  # seconds per epoch
            remaining_time = (epochs - epoch) * per_epoch_time
            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {train_loss / sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {eval_loss / validation_sourcing_periods:.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f} seconds"
                    f" - Est. Remaining time: {int(remaining_time)} seconds."
                )

        self.load_state_dict(best_state)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(f"Final best cost: {min_loss / sourcing_periods:.4f}")

    def reset(self) -> None:
        """
        Reset the controller to the initial state.
        """
        self.model = None
        self.sourcing_model = None


    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """Calculate the cost for the latest period."""
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

    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the total cost."""
        if seed is not None:
            torch.manual_seed(seed)

        
        total_cost = torch.tensor(0.0)
        for t in range(int(sourcing_periods/self.n_periods)):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()
            regular_q, expedited_q = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            # First pass for first time period in the cycle
            sourcing_model.order(regular_q, expedited_q[:, 0:1])
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()

            # Update inventory here - Not needed, handeld by the sourcing model
            # Order generates random demand and updates current inventory, past invetory, past regular and expedited orders
            
            for exp_q in expedited_q[:, 1:].split(1, dim=1):
                # Second pass for remaining time periods
                sourcing_model.order(
                    torch.zeros_like(exp_q),  # 0 regular
                    exp_q
                )
                last_cost = self.get_last_cost(sourcing_model)
                total_cost += last_cost.mean()

        return total_cost

    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average cost."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )
