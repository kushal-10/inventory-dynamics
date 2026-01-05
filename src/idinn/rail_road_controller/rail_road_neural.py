import logging
from datetime import datetime
from typing import List, Optional, no_type_check

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..rail_road_model import RailRoadInventoryModel

logger = logging.getLogger()


class RailRoadNeuralController(torch.nn.Module):
    """
    Neural controller for the Rail–Road inventory model using
    a shared trunk + deep cycle-specific heads.
    """

    def __init__(
        self,
        cycle_length: int,
        trunk_layers: List[int] = [64, 64],
        head_layers: List[int] = [32, 16],
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()

        self.model: Optional[RailRoadInventoryModel] = None
        self.cycle_length = cycle_length
        self.activation = activation

        # -----------------------
        # Shared trunk
        # -----------------------
        trunk = []
        in_dim = 1  # inventory only
        for h in trunk_layers:
            trunk.append(torch.nn.Linear(in_dim, h))
            trunk.append(self.activation)
            in_dim = h
        self.trunk = torch.nn.Sequential(*trunk)

        # -----------------------
        # Cycle-specific heads
        # -----------------------
        self.heads = torch.nn.ModuleList()
        for _ in range(cycle_length):
            head = []
            head_in = trunk_layers[-1]
            for h in head_layers:
                head.append(torch.nn.Linear(head_in, h))
                head.append(self.activation)
                head_in = h
            head.append(torch.nn.Linear(head_in, 1))
            head.append(torch.nn.ReLU())  # enforce non-negativity
            self.heads.append(torch.nn.Sequential(*head))

        logger.info(
            f"Initialized RailRoadNeuralController "
            f"(cycle_length={cycle_length}, trunk_layers={trunk_layers}, head_layers={head_layers})"
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inventory: torch.Tensor, cycle_day: int) -> torch.Tensor:
        """
        inventory: shape (batch, 1)
        cycle_day: int in {0, ..., K-1}
        """
        h = self.trunk(inventory)
        q = self.heads[cycle_day](h)
        q = q - torch.frac(q).detach()  # integerize like DP
        return q

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self) -> int:
        if self.model is None:
            raise RuntimeError("Controller not trained or model not attached.")

        inventory = self.model.get_current_inventory()
        cycle_day = self.model.get_current_cycle_day()

        q = self.forward(inventory, cycle_day)
        return int(q.item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @no_type_check
    def fit(
        self,
        model: RailRoadInventoryModel,
        periods: int,
        epochs: int,
        validation_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 100,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 3e-3,
        tensorboard_writer: Optional[SummaryWriter] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Train the neural controller by backpropagating through the environment.
        """
        self.model = model

        if seed is not None:
            torch.manual_seed(seed)

        optimizer_init_inventory = torch.optim.RMSprop(
            [model.init_inventory], lr=init_inventory_lr
        )
        optimizer_parameters = torch.optim.RMSprop(self.parameters(), lr=parameters_lr)

        min_loss = np.inf
        start_time = datetime.now()

        for epoch in tqdm(range(epochs)):
            optimizer_init_inventory.zero_grad()
            optimizer_parameters.zero_grad()

            model.reset(batch_size=1)
            total_cost = torch.zeros(1, requires_grad=True)

            for _ in range(periods):
                q = self.predict()  # int
                q_t = torch.tensor([[q]], dtype=torch.float32)

                model.order(q_t)

                I = model.get_current_inventory()  # tensor

                holding = model.get_holding_cost() * torch.relu(I)
                shortage = model.get_shortage_cost() * torch.relu(-I)

                if model.is_regular_day():
                    order_cost = model.get_regular_order_cost() * q_t
                else:
                    order_cost = model.get_expedited_order_cost() * q_t

                step_cost = order_cost + holding + shortage
                total_cost = total_cost + step_cost.mean()

            loss = total_cost / periods
            loss.backward()

            optimizer_init_inventory.step()
            optimizer_parameters.step()

            if loss < min_loss:
                min_loss = loss
                best_state = self.state_dict()

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("AvgCost/train", loss, epoch)

            if epoch % log_freq == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - Avg cost: {loss.item():.4f}"
                )
                print(loss)

        self.load_state_dict(best_state)

        logger.info(
            f"Training completed in {datetime.now() - start_time}, "
            f"best avg cost ≈ {min_loss:.4f}"
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.model = None
