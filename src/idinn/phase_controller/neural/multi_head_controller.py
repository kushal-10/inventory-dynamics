from typing import List, Optional, Tuple, Union
import logging
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseNeuralController
from ...sourcing_model import DualSourcingModel

logger = logging.getLogger()


class MultiPeriodNeuralControllerV2(torch.nn.Module, BaseNeuralController):
    """
    Multi-period dual-sourcing controller with two separate heads:

      - Even-period head (t = 0, 2, 4, ...): receives the full inventory state
        and outputs (Qr, Qe).
      - Odd-period head  (t = 1, 3, 5, ...): receives the *updated* inventory
        state (after one demand realization) and outputs (Qe,) only.

    The two heads share the same trunk (hidden layers) but have independent
    final linear layers, making it easy to have them specialise.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        n_periods: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        hidden_layers : list of int
            Number of neurons per hidden layer (shared trunk).
        activation : torch.nn.Module
            Activation function inserted between hidden layers.
        n_periods : int
            Number of time steps per cycle (must be > 1).
            Even steps → even head; odd steps → odd head.
        """
        super().__init__()

        assert n_periods > 1, "n_periods must be > 1"

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.n_periods = n_periods
        self.MAX_Q = 20

        # Populated by init_layers()
        self.trunk: Optional[torch.nn.Sequential] = None
        self.even_head: Optional[torch.nn.Linear] = None  # → (Qr, Qe)
        self.odd_head: Optional[torch.nn.Linear] = None   # → (Qe,)

        self.sourcing_model = None

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """Build shared trunk + two specialised heads."""

        input_length = regular_lead_time + expedited_lead_time + 1

        # ── Shared trunk ──────────────────────────────────────────────
        trunk_layers: List[torch.nn.Module] = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers) - 1):
            trunk_layers += [
                torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                self.activation,
            ]
        self.trunk = torch.nn.Sequential(*trunk_layers)

        last_hidden = self.hidden_layers[-1]

        # ── Even head: outputs [Qr, Qe] ──────────────────────────────
        self.even_head = torch.nn.Linear(last_hidden, 2)

        # ── Odd head: outputs [Qe] ────────────────────────────────────
        self.odd_head = torch.nn.Linear(last_hidden, 1)

        logger.info(
            f"Initialized MultiPeriodNeuralControllerV2 | "
            f"regular_lead_time={regular_lead_time}, "
            f"expedited_lead_time={expedited_lead_time}, "
            f"n_periods={self.n_periods}"
        )

    def _check_initialized(self) -> None:
        if self.trunk is None:
            raise AttributeError("Model not initialized. Call `init_layers()` first.")

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _apply_head(self, inputs: torch.Tensor, head: torch.nn.Linear) -> torch.Tensor:
        """Trunk → head → clamp → floor."""
        h = head(self.trunk(inputs))
        h = torch.clamp(h, 0.0, self.MAX_Q)
        # Straight-through floor so gradients still flow
        q = h - torch.frac(h).clone().detach()
        return q

    def forward_even(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Even-period forward pass.

        Returns
        -------
        regular_q  : (batch, 1)
        expedited_q: (batch, 1)
        """
        self._check_initialized()
        q = self._apply_head(inputs, self.even_head)
        return q[:, [0]], q[:, [1]]

    def forward_odd(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Odd-period forward pass.

        Returns
        -------
        expedited_q: (batch, 1)
        """
        self._check_initialized()
        return self._apply_head(inputs, self.odd_head)

    # ------------------------------------------------------------------
    # Public predict API  (mirrors original controller)
    # ------------------------------------------------------------------

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[List[int], torch.Tensor]],
        past_expedited_orders: Optional[Union[List[int], torch.Tensor]],
        period_index: int = 0,           # 0 = even, 1 = odd
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Predict order quantities for the given period parity.

        Parameters
        ----------
        period_index : int
            0 (or any even int) → even head → returns (Qr, Qe).
            1 (or any odd int)  → odd  head → returns (0,  Qe).
        output_tensor : bool
            If True return raw tensors, else return Python ints/floats.
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model,
        )

        if period_index % 2 == 0:
            regular_q, expedited_q = self.forward_even(inputs)
        else:
            expedited_q = self.forward_odd(inputs)
            regular_q = torch.zeros_like(expedited_q)

        if output_tensor:
            return regular_q, expedited_q
        else:
            return int(regular_q.item()), int(expedited_q.item())

    # ------------------------------------------------------------------
    # Cost helpers  (unchanged from original)
    # ------------------------------------------------------------------

    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = (
            sourcing_model.get_regular_order_cost() * last_regular_q
            + sourcing_model.get_expedited_order_cost() * last_expedited_q
            + sourcing_model.get_holding_cost() * torch.relu(current_inventory)
            + sourcing_model.get_shortage_cost() * torch.relu(-current_inventory)
        )
        return last_cost

    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Roll out the model for `sourcing_periods` steps, alternating heads.

        Cycle layout (n_periods=2 example):
            t=0  even head → place Qr + Qe   (regular pipeline starts)
            t=1  odd  head → place 0  + Qe   (expedited only)
        """
        total_cost = torch.tensor(0.0)

        for t in range(sourcing_periods):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()

            inputs = self.prepare_inputs(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                sourcing_model=sourcing_model,
            )

            if t % self.n_periods == 0:
                # Even step: decide both Qr and Qe
                regular_q, expedited_q = self.forward_even(inputs)
            else:
                # Odd step: decide only Qe, no regular replenishment
                expedited_q = self.forward_odd(inputs)
                regular_q = torch.zeros_like(expedited_q)

            sourcing_model.order(regular_q, expedited_q)
            total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return self.get_total_cost(sourcing_model, sourcing_periods, seed) / sourcing_periods

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 10,
        init_inventory_freq: int = 4,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 3e-3,
        seed: Optional[int] = None,
    ) -> None:
        """Train the controller. API is identical to the original."""
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.trunk is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )

        start_time = datetime.now()
        logger.info(f"Starting MultiPeriodNeuralControllerV2 training at {start_time}")
        logger.info(
            f"Sourcing model: batch_size={sourcing_model.batch_size}, "
            f"lead_time={sourcing_model.lead_time}, "
            f"init_inventory={sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training: epochs={epochs}, sourcing_periods={sourcing_periods}, "
            f"validation_periods={validation_sourcing_periods}, lr={parameters_lr}"
        )

        optimizer_inventory = torch.optim.RMSprop(
            [sourcing_model.init_inventory], lr=init_inventory_lr
        )
        optimizer_params = torch.optim.Adam(self.parameters(), lr=parameters_lr)

        min_loss = np.inf
        best_state = self.state_dict()

        for epoch in tqdm(range(epochs)):
            optimizer_inventory.zero_grad()
            optimizer_params.zero_grad()

            sourcing_model.reset()
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods, seed=seed)
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)

            if epoch % init_inventory_freq == 0:
                optimizer_inventory.step()
            else:
                optimizer_params.step()

            # ── Validation & checkpointing ────────────────────────────
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_loss = self.get_average_cost(sourcing_model, validation_sourcing_periods)
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = self.state_dict()
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_state = self.state_dict()

            # ── Logging ───────────────────────────────────────────────
            duration = (datetime.now() - start_time).total_seconds()
            per_epoch_time = duration / (epoch + 1)
            remaining = (epochs - epoch - 1) * per_epoch_time

            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Train cost: {train_loss / sourcing_periods:.4f}"
                    f" - {per_epoch_time:.2f}s/epoch"
                    f" - ETA: {int(remaining)}s"
                )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs}"
                    f" - Val cost: {eval_loss / validation_sourcing_periods:.4f}"
                )

        self.load_state_dict(best_state)

        end_time = datetime.now()
        logger.info(f"Training completed at {end_time} | duration: {end_time - start_time}")
        logger.info(f"Best cost: {min_loss / sourcing_periods:.4f}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_regular_orders: torch.Tensor,
        past_expedited_orders: torch.Tensor,
        sourcing_model: DualSourcingModel,
    ) -> torch.Tensor:
        """Identical input preparation logic as the original controller."""
        regular_lead_time = sourcing_model.get_regular_lead_time()
        expedited_lead_time = sourcing_model.get_expedited_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(past_regular_orders, regular_lead_time)
        past_expedited_orders = self._check_past_orders(past_expedited_orders, expedited_lead_time)

        if expedited_lead_time > 0:
            inputs = torch.cat(
                [current_inventory, past_expedited_orders[:, -expedited_lead_time:]], dim=1
            )
        else:
            inputs = current_inventory

        if regular_lead_time > 0:
            inputs = torch.cat([inputs, past_regular_orders[:, -regular_lead_time:]], dim=1)

        return inputs

    def reset(self) -> None:
        self.trunk = None
        self.even_head = None
        self.odd_head = None
        self.sourcing_model = None