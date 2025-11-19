# src/idinn/dual_neural/triple_neural.py
from typing import Optional, Tuple, List
import torch
import numpy as np
from datetime import datetime


# ---------- helpers ----------
def ste_floor(a: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for floor(a).
    Forward: floor(a); backward: gradient of identity.
    """
    return a + (torch.floor(a) - a).detach()


class LogReLU(torch.nn.Module):
    def __init__(self, beta_value: float = 0.5):
        super().__init__()
        self.beta_value = float(beta_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(self.beta_value * torch.relu(x) + 1.0, min=1e-9))


# ---------- Controller ----------
class DualSourcingNeuralController(torch.nn.Module):
    """
    Controller adapted for the 3-day cycle model:
    - Input: current inventory (batch,1)
    - Output (raw): three values [qe1, qe2, qr]
    - After positive transform + STE floor, returns integer-like tensors (batch,1).
    - fit(...) implements training loop (BPTT through cycles via env.order()).
    """

    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: torch.nn.Module = torch.nn.CELU(alpha=1.0),
        use_ste: bool = True,
        expedited_activation: torch.nn.Module = torch.nn.ReLU(),
        regular_activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.use_ste = bool(use_ste)
        self.expedited_activation = expedited_activation
        self.regular_activation = regular_activation

        # build MLP: input_dim = 1 (I_t)
        layers = []
        in_dim = 1
        for h in self.hidden_layers:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(self.activation)
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, 3))  # outputs: qe1, qe2, qr
        self.net = torch.nn.Sequential(*layers)

        # bookkeeping
        self.sourcing_model = None

    def forward(self, I_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        I_t: (batch,1)
        returns: qe1, qe2, qr as tensors (batch,1)
        """
        raw = self.net(I_t)  # (batch,3)
        # apply smooth positive transform (softplus) for stability
        pos = torch.nn.functional.softplus(raw)

        qe1 = pos[:, [0]]
        qe2 = pos[:, [1]]
        qr = pos[:, [2]]

        # optional custom activations (kept for API compatibility)
        # expedite/regular activations could be modules; apply if they are callable
        if isinstance(self.expedited_activation, torch.nn.Module):
            qe1 = self.expedited_activation(qe1)
            qe2 = self.expedited_activation(qe2)
        if isinstance(self.regular_activation, torch.nn.Module):
            qr = self.regular_activation(qr)

        if self.use_ste:
            qe1 = ste_floor(qe1)
            qe2 = ste_floor(qe2)
            qr = ste_floor(qr)

        return qe1, qe2, qr

    def predict(
        self,
        current_inventory: torch.Tensor,
        output_tensor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (regular_q, expedited_q).
        expedited_q returned = qe1 + qe2 (so compatible if external code expects aggregated expedited).
        If output_tensor=False and batch_size==1, returns python ints (reg, exp).
        """
        if self.sourcing_model is None:
            raise AttributeError("Controller has no associated sourcing_model (set during fit).")

        I_t = current_inventory
        if isinstance(I_t, torch.Tensor) is False:
            I_t = torch.tensor([[float(I_t)]], dtype=torch.float32)

        qe1, qe2, qr = self.forward(I_t)
        expedited = qe1 + qe2
        if output_tensor:
            # return tensors shaped (batch,1): regular, expedited
            return qr, expedited
        # else: return ints if batch == 1
        if qr.shape[0] != 1:
            raise ValueError("output_tensor=False only allowed for batch_size==1")
        return int(qr.detach().cpu().item()), int(expedited.detach().cpu().item())

    def fit(
        self,
        sourcing_model,
        sourcing_periods: int,
        epochs: int,
        validation_sourcing_periods: Optional[int] = None,
        validation_freq: int = 50,
        log_freq: int = 100,
        init_inventory_lr: float = 1e-1,
        parameters_lr: float = 3e-3,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Simple training loop. The controller will store a reference to sourcing_model.
        The sourcing_model must implement:
        - reset(batch_size)
        - get_current_inventory()
        - order(qe1, qe2, qr) -> returns cycle_cost (batch,1)
        - init_inventory tensor (trainable if you pass it to optimizer)
        """
        device = device or torch.device("cpu")
        self.to(device)
        self.sourcing_model = sourcing_model
        sourcing_model.init_inventory = sourcing_model.init_inventory.to(device)

        if seed is not None:
            torch.manual_seed(seed)

        # optimizer with two param groups: model params + init_inventory
        optimizer = torch.optim.RMSprop(
            [
                {"params": self.parameters(), "lr": parameters_lr},
                {"params": [sourcing_model.init_inventory], "lr": init_inventory_lr},
            ]
        )

        best_state = self.state_dict()
        min_loss = float("inf")
        start_time = datetime.now()

        for epoch in range(epochs):
            optimizer.zero_grad()
            sourcing_model.reset(batch_size=sourcing_model.batch_size)
            total_cycle_cost = 0.0

            for t in range(sourcing_periods):
                I_t = sourcing_model.get_current_inventory().to(device)
                qe1, qe2, qr = self.forward(I_t)
                cycle_cost = sourcing_model.order(qe1, qe2, qr)  # (batch,1)
                total_cycle_cost = total_cycle_cost + cycle_cost.mean()

            loss = total_cycle_cost / sourcing_periods
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            # validation if requested
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                with torch.no_grad():
                    sourcing_model.reset(batch_size=sourcing_model.batch_size)
                    val_total = 0.0
                    for t in range(validation_sourcing_periods):
                        I_t = sourcing_model.get_current_inventory().to(device)
                        qe1, qe2, qr = self.forward(I_t)
                        val_total += sourcing_model.order(qe1, qe2, qr).mean()
                    val_loss = val_total / validation_sourcing_periods
                    if val_loss < min_loss:
                        min_loss = float(val_loss.detach().cpu().item())
                        best_state = self.state_dict()
            else:
                if float(loss.detach().cpu().item()) < min_loss:
                    min_loss = float(loss.detach().cpu().item())
                    best_state = self.state_dict()

            if epoch % log_freq == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"[{epoch}/{epochs}] loss={loss.item():.4f} avg_per_cycle={loss.item():.4f} elapsed_s={elapsed:.1f}")

        # load best
        self.load_state_dict(best_state)
        print(f"Training finished. Best avg cycle cost: {min_loss:.4f}")
        return

    # simple simulation helper for plotting
    def simulate(self, sourcing_model, cycles: int = 50, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        sourcing_model.reset(batch_size=1)
        I_hist = []
        qe1_hist = []
        qe2_hist = []
        qr_hist = []
        dem_hist = []
        for t in range(cycles):
            I_t = sourcing_model.get_current_inventory()
            qe1, qe2, qr = self.forward(I_t)
            sourcing_model.order(qe1, qe2, qr)
        return sourcing_model.get_histories_numpy()
