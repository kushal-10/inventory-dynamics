import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..sourcing_model import DualSourcingModel
from .base import BasePeriodicDualController

# Get root logger
logger = logging.getLogger()




class MultiHeadedNeuralController(torch.nn.Module, BasePeriodicDualController):
    """
    MultiHeadedNeuralController is a neural network controller for dual sourcing inventory optimization with periodic restrictions.
    """

    def __init__(
        self,
        shared_layers: List[int] = [128, 64],
        head_restricted_layers: List[int] = [32, 16],
        head_regular_layers: List[int] = [32],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        MAX_Q: int = 20, # Hard cap on order quantities
        #TODO: calculate the same way used in DP
        compressed: bool = False,
    ):
        super().__init__()
        self.sourcing_model = None

        self.shared_layers = shared_layers
        self.head_regular_layers = head_regular_layers
        self.head_restricted_layers = head_restricted_layers

        self.activation = activation
        self.compressed = compressed
        self.MAX_Q = MAX_Q

        self.shared: Optional[torch.nn.Sequential] = None
        self.head_regular: Optional[torch.nn.Sequential] = None
        self.head_restricted: Optional[torch.nn.Sequential] = None

        logger.info(
            f"Initialized PeriodicNaiveNeuralController "
            f"(shared={shared_layers}, even={head_regular_layers}, odd={head_restricted_layers})"
        )


    def _build_mlp(self, in_dim: int, layers: List[int], out_dim: int) -> torch.nn.Sequential:
        modules = []
        prev = in_dim
        for h in layers:
            modules.append(torch.nn.Linear(prev, h))
            modules.append(self.activation)
            prev = h
        modules.append(torch.nn.Linear(prev, out_dim))
        return torch.nn.Sequential(*modules)


    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Initialize the layers of the neural network.

        Parameters
        ----------
        regular_lead_time : int
            Regular lead time.
        expedited_lead_time : int
            Expedited lead time.
        """

        
        if self.compressed:
            # add +1 to handle phase 
            input_length = regular_lead_time + expedited_lead_time + 1 
        else:
            input_length = regular_lead_time + expedited_lead_time + 2

        # Shared trunk
        self.shared = self._build_mlp(
            in_dim=input_length,
            layers=self.shared_layers,
            out_dim=self.shared_layers[-1],
        )

        shared_out_dim = self.shared_layers[-1]

        # Heads
        self.head_regular = self._build_mlp(
            in_dim=shared_out_dim,
            layers=self.head_regular_layers,
            out_dim=2,
        )

        self.head_restricted = self._build_mlp(
            in_dim=shared_out_dim,
            layers=self.head_restricted_layers,
            out_dim=1,
        )

        logger.info(
            f"Initialized two-headed network "
            f"(input={input_length}, shared_out={shared_out_dim})"
        )

    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        phase: int,
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

        if regular_lead_time > 0:
            if self.compressed:
                inputs = past_regular_orders[:, -regular_lead_time:]
                inputs[:, 0] += current_inventory
            else:
                inputs = torch.cat(
                    [
                        current_inventory,
                        past_regular_orders[:, -regular_lead_time:],
                    ],
                    dim=1,
                )
        else:
            inputs = current_inventory

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_expedited_orders[:, -expedited_lead_time:]], dim=1
            )


        phase_tensor = torch.full(
            (inputs.shape[0], 1),
            float(phase),
            dtype=inputs.dtype,
        )

        inputs = torch.cat([inputs, phase_tensor], dim=1)
        
        return inputs

    def forward(self, inputs: torch.Tensor, phase: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shared is None or self.head_regular is None or self.head_restricted is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")

        h_shared = self.shared(inputs)

        if phase is not None:

            if phase > 0: # Restricted phases - 1,2,3....
                h = self.head_restricted(h_shared)

                qe = h[:, [0]] 
        
                h = torch.cat([qe], dim=1)

                # hard cap
                h = torch.clamp(h, 0.0, self.MAX_Q)

                q = h - torch.frac(h).clone().detach()
                expedited_q = q[:, [0]]
                return expedited_q
                
            else:
                h = self.head_regular(h_shared)

                qr = h[:, [0]] 
                qe = h[:, [1]]
        
                h = torch.cat([qr, qe], dim=1)

                # hard cap
                h = torch.clamp(h, 0.0, self.MAX_Q)

                q = h - torch.frac(h).clone().detach()
                regular_q = q[:, [0]]
                expedited_q = q[:, [1]]
                return regular_q, expedited_q

        

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        phase: Optional[int] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        f"""
        Forward pass of the neural network.

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
        phase : int, the current phase of the input - {0,1}
        Returns
        -------
        tuple
            A tuple containing the regular order quantity and expedited order quantity.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            phase,
            sourcing_model=self.sourcing_model,
        )
        regular_q, expedited_q = self.forward(inputs, phase)

        if output_tensor:
            return regular_q, expedited_q
        else:
            return int(regular_q), int(expedited_q)

    @no_type_check
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
        weight_decay_shared: float = 0.0,
        weight_decay_odd: float = 0.0,
        weight_decay_even: float = 0.0,
        parameters_lr_shared: float = 1e-3,
        parameters_lr_odd: float = 1e-3,
        parameters_lr_even: float = 1e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
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
        tensorboard_writer : tensorboard.SummaryWriter, optional
        seed : int, optional
            Random seed for reproducibility.
            Tensorboard writer for logging.
        """
        # Store sourcing model in self.sourcing_model
    

        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.shared is None or self.head_regular is None or self.head_restricted is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        start_time = datetime.now()
        logger.info(f"Starting dual sourcing neural network training at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"validation_periods={validation_sourcing_periods}"
        )

        
        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        
        optimizer_parameters = torch.optim.RMSprop(
            [
                {
                    "params": self.shared.parameters(),
                    "lr": parameters_lr_shared,
                    "weight_decay": weight_decay_shared,
                },
                {
                    "params": self.head_restricted.parameters(),
                    "lr": parameters_lr_odd,
                    "weight_decay": weight_decay_odd,
                },
                {
                    "params": self.head_regular.parameters(),
                    "lr": parameters_lr_even,
                    "weight_decay": weight_decay_even,
                },
            ]
        )

        min_loss = np.inf

        for epoch in tqdm(range(epochs)):
            # Clear grad cache
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            # Reset the sourcing model with the learned init inventory
            sourcing_model.reset() # clean
            train_loss = super().get_periodic_total_cost(sourcing_model, sourcing_periods)
            train_loss.backward()
            # Perform gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_loss = super().get_periodic_total_cost(
                    sourcing_model, validation_sourcing_periods
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
        self.shared = None
        self.head_regular = None
        self.head_restricted = None
        self.sourcing_model = None
