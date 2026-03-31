import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ..sourcing_model import DualSourcingModel
from .base import BaseDPController

# Get root logger
logger = logging.getLogger()


class CyclicDualSourcingNeuralController(torch.nn.Module, BaseDPController):
    """
    Neural network controller for cyclic dual-sourcing inventory optimization.

    Places a regular replenishment order once every N periods (the cycle), and an
    expedited order every period.  The network observes the current inventory and
    pipeline state and simultaneously outputs all N+1 order quantities for the
    coming cycle: (qr0, qe0, qe1, ..., qe_{N-1}).

    Parameters
    ----------
    cycle_length : int, default is 2
        Number of periods per cycle (1, 2, or 3).  A regular order is placed only
        in period 0 of each cycle; expedited orders are placed every period.
    hidden_layers : list of int, default is [128, 64, 32, 16, 8, 4]
        Sizes of the hidden layers.
    activation : torch.nn.Module, default is torch.nn.ReLU()
        Activation function used between hidden layers.
    compressed : bool, default is False
        If True, the current inventory is folded into the first pipeline position
        rather than passed as a separate input feature.

    Attributes
    ----------
    cycle_length : int
    hidden_layers : list of int
    activation : torch.nn.Module
    compressed : bool
    model : torch.nn.Sequential or None
        Initialized by :meth:`init_layers`.
    sourcing_model : DualSourcingModel or None
        Set during :meth:`fit`.
    """

    def __init__(
        self,
        cycle_length: int = 2,
        hidden_layers: List[int] = [128, 64, 32, 16, 8, 4],
        activation: torch.nn.Module = torch.nn.ReLU(),
        compressed: bool = False,
    ) -> None:
        super().__init__()
        if cycle_length not in (1, 2, 3):
            raise ValueError("cycle_length must be 1, 2, or 3")
        self.cycle_length = cycle_length
        self.sourcing_model = None
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed
        self.model: Optional[torch.nn.Sequential] = None
        logger.info(
            f"Initialized CyclicDualSourcingNeuralController with "
            f"cycle_length={cycle_length}, hidden_layers={hidden_layers}, compressed={compressed}"
        )

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Initialize the neural network layers.

        Parameters
        ----------
        regular_lead_time : int
        expedited_lead_time : int
        """
        if self.compressed:
            input_length = regular_lead_time + expedited_lead_time
        else:
            input_length = regular_lead_time + expedited_lead_time + 1

        # Output size: qr0 + qe0 .. qe_{N-1} = cycle_length + 1 values
        output_size = self.cycle_length + 1

        architecture = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers) - 1):
            architecture += [
                torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                self.activation,
            ]
        architecture += [
            torch.nn.Linear(self.hidden_layers[-1], output_size),
            torch.nn.ReLU(),  # All orders must be non-negative
        ]
        self.model = torch.nn.Sequential(*architecture)
        logger.info(
            f"Initialized network layers: input={input_length}, "
            f"output={output_size} (cycle_length+1), "
            f"regular_lead_time={regular_lead_time}, expedited_lead_time={expedited_lead_time}"
        )

    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        sourcing_model: DualSourcingModel,
    ) -> torch.Tensor:
        """Build the input tensor from inventory state and pipeline orders."""
        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(past_regular_orders, regular_lead_time)
        past_expedited_orders = self._check_past_orders(past_expedited_orders, expedited_lead_time)

        if regular_lead_time > 0:
            if self.compressed:
                inputs = past_regular_orders[:, -regular_lead_time:].clone()
                inputs[:, 0] += current_inventory.squeeze(1)
            else:
                inputs = torch.cat(
                    [current_inventory, past_regular_orders[:, -regular_lead_time:]],
                    dim=1,
                )
        else:
            inputs = current_inventory

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [inputs, past_expedited_orders[:, -expedited_lead_time:]], dim=1
            )
        return inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor of shape (batch, input_length)

        Returns
        -------
        tuple of torch.Tensor
            (qr0, qe0, qe1, ..., qe_{N-1}), each of shape (batch, 1).
            Values are floored to integers (gradient flows through the floor).
        """
        if self.model is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")
        h = self.model(inputs)
        q = h - torch.frac(h).clone().detach()
        return tuple(q[:, [i]] for i in range(self.cycle_length + 1))

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[int, ...]]:
        """
        Predict order quantities for the next cycle.

        Parameters
        ----------
        current_inventory : int or torch.Tensor
        past_regular_orders : list or torch.Tensor, optional
        past_expedited_orders : list or torch.Tensor, optional
        output_tensor : bool, default is False
            If True, returns a tuple of torch.Tensors.  Otherwise integers.

        Returns
        -------
        tuple
            (qr0, qe0, qe1, ..., qe_{N-1}) — length cycle_length + 1.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model,
        )
        orders = self.forward(inputs)

        if output_tensor:
            return orders
        return tuple(int(q) for q in orders)

    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """Calculate the cost for the latest single period."""
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        return (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Accumulate cost over ``sourcing_periods`` cycles.

        Each cycle covers ``cycle_length`` real periods.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
        sourcing_periods : int
            Number of cycles to simulate.
        seed : int, optional
        """
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()

            # orders = (qr0, qe0, qe1, ..., qe_{N-1})
            orders = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            # Period 0: regular + first expedited
            sourcing_model.order(orders[0], orders[1])
            total_cost += self.get_last_cost(sourcing_model).mean()

            # Periods 1..N-1: expedited only
            for t in range(1, self.cycle_length):
                sourcing_model.order(torch.zeros_like(orders[0]), orders[t + 1])
                total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Average per-period cost over ``sourcing_periods`` cycles."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / (sourcing_periods * self.cycle_length)
        )

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
        parameters_lr: float = 3e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Train the neural network controller.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
        sourcing_periods : int
            Number of cycles per training epoch.
        epochs : int
        validation_sourcing_periods : int, optional
        validation_freq : int, default is 50
            Run validation every this many epochs.
        log_freq : int, default is 100
            Log training cost every this many epochs.
        init_inventory_freq : int, default is 4
            Update initial inventory every this many epochs; otherwise update
            network parameters.
        init_inventory_lr : float, default is 1e-1
        parameters_lr : float, default is 3e-3
        tensorboard_writer : SummaryWriter, optional
        seed : int, optional
        """
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.model is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        start_time = datetime.now()
        logger.info(f"Starting cyclic dual sourcing neural network training at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={sourcing_model.batch_size}, "
            f"lead_time={sourcing_model.lead_time}, "
            f"init_inventory={sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"cycle_length={self.cycle_length}, "
            f"validation_periods={validation_sourcing_periods}, learning_rate={parameters_lr}"
        )

        optimizer_init_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
        min_loss = np.inf
        best_state = self.state_dict()
        best_init_inventory = sourcing_model.init_inventory.data.clone()

        for epoch in range(epochs):
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()
            sourcing_model.reset()
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods)

            if torch.isnan(train_loss):
                logger.warning(
                    f"Epoch {epoch}: NaN loss detected, restoring best state and resetting optimizers"
                )
                self.load_state_dict(best_state)
                sourcing_model.init_inventory.data.copy_(best_init_inventory)
                optimizer_init_inventory = torch.optim.RMSprop(
                    [sourcing_model.init_inventory], lr=init_inventory_lr
                )
                optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)
                continue

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([sourcing_model.init_inventory], max_norm=1.0)

            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_loss = self.get_total_cost(sourcing_model, validation_sourcing_periods)
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = self.state_dict()
                    best_init_inventory = sourcing_model.init_inventory.data.clone()
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_state = self.state_dict()
                    best_init_inventory = sourcing_model.init_inventory.data.clone()

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train",
                    train_loss / (sourcing_periods * self.cycle_length),
                    epoch,
                )
                if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_loss / (validation_sourcing_periods * self.cycle_length),
                        epoch,
                    )
                tensorboard_writer.flush()

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)
            remaining_time = (epochs - epoch) * per_epoch_time

            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Training cost: {train_loss / (sourcing_periods * self.cycle_length):.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f}s"
                    f" - Est. remaining: {int(remaining_time)}s"
                )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Validation cost: {eval_loss / (validation_sourcing_periods * self.cycle_length):.4f}"
                    f" - Per epoch time: {per_epoch_time:.2f}s"
                    f" - Est. remaining: {int(remaining_time)}s"
                )

        self.load_state_dict(best_state)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(
            f"Final best cost: {min_loss / (sourcing_periods * self.cycle_length):.4f}"
        )

    def reset(self) -> None:
        """Reset the controller to its initial (untrained) state."""
        self.model = None
        self.sourcing_model = None

    def save(self, path: str) -> None:
        """Save the model weights to ``path``."""
        torch.save(self.model, path)

    def load(self, path: str) -> None:
        """Load model weights from ``path``."""
        self.model = torch.load(path, weights_only=False)
