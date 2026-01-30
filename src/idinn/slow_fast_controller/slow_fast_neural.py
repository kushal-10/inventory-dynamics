import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..slow_fast import CyclicSlowFastModel
from .base import BaseSlowFastController

logger = logging.getLogger()


class CyclicSlowFastNeuralController(torch.nn.Module, BaseSlowFastController):
    """
    Neural controller for cyclic slow–fast sourcing with three heads:
    sr0, se0, sr1.
    """

    def __init__(
        self,
        hidden_layers: List[int] = [128, 64],
        activation: torch.nn.Module = torch.nn.ReLU(),
        compressed: bool = False,
        cycle_length: int = 2,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed
        self.cycle_length = cycle_length

        self.encoder = None
        self.head_sr0 = None
        self.head_se0 = None
        self.head_sr1 = None

        self.sourcing_model: Optional[CyclicSlowFastModel] = None
        self._t: int = 0  # internal time counter

    # ------------------------------------------------------------------
    # Network initialization
    # ------------------------------------------------------------------

    def init_layers(self, slow_lead_time: int, fast_lead_time: int) -> None:
        if self.compressed:
            input_length = slow_lead_time + fast_lead_time + 1
        else:
            input_length = slow_lead_time + fast_lead_time + 2

        layers = [
            torch.nn.Linear(input_length, self.hidden_layers[0]),
            self.activation,
        ]
        for i in range(len(self.hidden_layers) - 1):
            layers += [
                torch.nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]),
                self.activation,
            ]

        self.encoder = torch.nn.Sequential(*layers)

        self.head_sr0 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.ReLU(),
        )
        self.head_se0 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.ReLU(),
        )
        self.head_sr1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.ReLU(),
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        current_inventory: torch.Tensor,
        past_slow_orders: torch.Tensor,
        past_fast_orders: torch.Tensor,
        t: int,
        sourcing_model: CyclicSlowFastModel,
    ) -> torch.Tensor:

        slow_lt = sourcing_model.get_slow_lead_time()
        fast_lt = sourcing_model.get_fast_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_slow_orders = self._check_past_orders(past_slow_orders, slow_lt)
        past_fast_orders = self._check_past_orders(past_fast_orders, fast_lt)

        if slow_lt > 0:
            if self.compressed:
                inputs = past_slow_orders[:, -slow_lt:]
                inputs[:, 0] += current_inventory
            else:
                inputs = torch.cat(
                    [current_inventory, past_slow_orders[:, -slow_lt:]],
                    dim=1,
                )
        else:
            inputs = current_inventory

        if fast_lt > 0:
            inputs = torch.cat(
                [inputs, past_fast_orders[:, -fast_lt:]], dim=1
            )

        tau = (t % self.cycle_length) / self.cycle_length
        tau_tensor = torch.full_like(current_inventory, tau)

        return torch.cat([inputs, tau_tensor], dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor):
        h = self.encoder(inputs)

        sr0 = self.head_sr0(h)
        se0 = self.head_se0(h)
        sr1 = self.head_sr1(h)

        sr0 = sr0 - torch.frac(sr0).detach()
        se0 = se0 - torch.frac(se0).detach()
        sr1 = sr1 - torch.frac(sr1).detach()

        return sr0, se0, sr1

    # ------------------------------------------------------------------
    # Prediction with internal clock
    # ------------------------------------------------------------------

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_slow_orders: Optional[Union[List[int], torch.Tensor]] = None,
        past_fast_orders: Optional[Union[List[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ):

        if self.sourcing_model is None:
            raise AttributeError("Controller is not trained.")

        t = self._t
        self._t += 1

        inputs = self.prepare_inputs(
            current_inventory,
            past_slow_orders,
            past_fast_orders,
            t,
            self.sourcing_model,
        )

        sr0, se0, sr1 = self.forward(inputs)

        if t % self.cycle_length == 0:
            slow_q, fast_q = sr0, se0
        else:
            slow_q = sr1
            fast_q = torch.zeros_like(sr1)

        if output_tensor:
            return slow_q, fast_q
        else:
            return int(slow_q), int(fast_q)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @no_type_check
    def fit(
        self,
        sourcing_model: CyclicSlowFastModel,
        sourcing_periods: int,
        epochs: int,
        parameters_lr: float = 3e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
        seed: Optional[int] = None,
    ) -> None:

        self.sourcing_model = sourcing_model
        self._t = 0

        if seed is not None:
            torch.manual_seed(seed)

        if self.encoder is None:
            self.init_layers(
                sourcing_model.get_slow_lead_time(),
                sourcing_model.get_fast_lead_time(),
            )

        optimizer = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)

        min_loss = np.inf
        best_state = None

        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            sourcing_model.reset()
            self._t = 0

            loss = self.get_total_cost(sourcing_model, sourcing_periods)
            logger.info(f"Epoch {epoch+1}/{epochs}, loss: {loss/sourcing_periods:.2f}")
            loss.backward()
            optimizer.step()

            if loss < min_loss:
                min_loss = loss
                best_state = self.state_dict()


        if best_state is not None:
            self.load_state_dict(best_state)
            logger.info(f"Best loss {min_loss/sourcing_periods:.2f}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.encoder = None
        self.head_sr0 = None
        self.head_se0 = None
        self.head_sr1 = None
        self.sourcing_model = None
        self._t = 0
