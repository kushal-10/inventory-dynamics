# three_day_cycle.py
from typing import Optional, Tuple
import torch
import numpy as np


# ---------------------------
# Helpers: STE and LogReLU
# ---------------------------
def ste_floor(a: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for floor(a).
    Forward: floor(a), Backward: gradient of identity on a.
    Implemented as: a + (floor(a) - a).detach()
    """
    return a + (torch.floor(a) - a).detach()


class LogReLU(torch.nn.Module):
    """
    Vectorized LogReLU:
      out = log(beta * relu(x) + 1)
    Keeps autograd & device.
    """
    def __init__(self, beta_value: float = 0.5):
        super().__init__()
        self.beta_value = float(beta_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(self.beta_value * torch.relu(x) + 1.0, min=1e-9))


# ---------------------------
# Environment: ThreeDayCycleModel
# ---------------------------
class ThreeDayCycleModel:
    """
    Three-day cycle inventory environment.

    Orders arrive same day (no lead time).
    The controller issues q_e1 (expedited day 1), q_e2 (expedited day 2), q_r (regular on day 3).
    Inventory updates across three subperiods and the cycle cost is the sum of subperiod costs.

    demand_generator must implement `.sample(batch_size)` returning a tensor shaped (batch, 1)
    or you can call it three times if it only returns (batch,1) per call.
    """

    def __init__(
        self,
        holding_cost: float,
        shortage_cost: float,
        init_inventory: float,
        demand_generator,
        batch_size: int = 1,
        regular_order_cost: float = 0.0,
        expedited_order_cost: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.holding_cost = float(holding_cost)
        self.shortage_cost = float(shortage_cost)
        self.regular_order_cost = float(regular_order_cost)
        self.expedited_order_cost = float(expedited_order_cost)

        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.demand_generator = demand_generator

        # trainable initial inventory if desired
        self.init_inventory = torch.tensor([float(init_inventory)], requires_grad=True, dtype=torch.float32).to(self.device)

        # internal histories (we keep them for simulation/plotting)
        self.reset(batch_size=self.batch_size)

    def reset(self, batch_size: Optional[int] = None):
        if batch_size is not None:
            self.batch_size = batch_size

        # past_inventories stores the sequence of inventories at cycle boundaries and subperiods if needed.
        # We'll start with the initial inventory repeated for batch.
        I0 = (self.init_inventory - torch.frac(self.init_inventory).clone().detach()).to(self.device)
        self.past_inventories = I0.repeat(self.batch_size, 1)  # shape (batch, 1)
        # We'll keep per-cycle record lists too:
        self._history_I1 = []  # list of tensors (batch,1) for subperiod 1
        self._history_I2 = []
        self._history_I3 = []
        self._history_qe1 = []
        self._history_qe2 = []
        self._history_qr = []
        self._history_demands = []  # list of (batch,3) demands

    def get_current_inventory(self) -> torch.Tensor:
        # return starting inventory for next cycle (shape (batch, 1))
        return self.past_inventories[:, [-1]]

    def sample_three_demands(self) -> torch.Tensor:
        """
        Attempts to sample three demands from demand_generator.
        If demand_generator.sample supports vectorized `n`, use it; otherwise call it 3 times.
        Returns: tensor shape (batch, 3)
        """
        # Try to call sample(batch, n=3) if available
        sample_fn = getattr(self.demand_generator, "sample", None)
        if sample_fn is None:
            raise AttributeError("demand_generator must implement .sample(batch_size)")

        # try vectorized interface: sample(batch, periods)
        try:
            maybe = sample_fn(self.batch_size, periods=3)  # custom interface if exists
            if isinstance(maybe, torch.Tensor) and maybe.shape == (self.batch_size, 3):
                return maybe.to(self.device).float()
        except TypeError:
            pass

        # fallback: call sample() three times (common case)
        d1 = sample_fn(self.batch_size)
        d2 = sample_fn(self.batch_size)
        d3 = sample_fn(self.batch_size)
        # ensure tensors and shape
        def _to_col(x):
            if isinstance(x, torch.Tensor):
                return x.reshape(self.batch_size, 1).to(self.device).float()
            else:
                return torch.tensor(x).reshape(self.batch_size, 1).to(self.device).float()
        d1 = _to_col(d1)
        d2 = _to_col(d2)
        d3 = _to_col(d3)
        return torch.cat([d1, d2, d3], dim=1)  # (batch,3)

    def order(self, qe1: torch.Tensor, qe2: torch.Tensor, qr: torch.Tensor):
        """
        Apply one cycle's orders and demands; update histories.

        qe1, qe2, qr: tensors of shape (batch,1) (can be float tensors — STE must be applied in controller)
        Returns: cost tensor (batch,1) for this cycle.
        """
        # make sure on right device and shape
        qe1 = qe1.to(self.device).reshape(self.batch_size, 1)
        qe2 = qe2.to(self.device).reshape(self.batch_size, 1)
        qr  = qr.to(self.device).reshape(self.batch_size, 1)

        I_t = self.get_current_inventory()  # (batch,1)
        demands = self.sample_three_demands()  # (batch,3)
        d1 = demands[:, [0]]
        d2 = demands[:, [1]]
        d3 = demands[:, [2]]

        # subperiod inventories
        I1 = I_t + qe1 - d1
        I2 = I1  + qe2 - d2
        I3 = I2  + qr  - d3  # this becomes I_{t+1}

        # store histories
        self._history_I1.append(I1)
        self._history_I2.append(I2)
        self._history_I3.append(I3)
        self._history_qe1.append(qe1)
        self._history_qe2.append(qe2)
        self._history_qr.append(qr)
        self._history_demands.append(demands)

        # append I3 as the next cycle boundary inventory
        self.past_inventories = torch.cat([self.past_inventories, I3], dim=1)

        # compute cycle cost as specified
        holding = self.holding_cost * (torch.relu(I1) + torch.relu(I2) + torch.relu(I3))
        shortage = self.shortage_cost * (torch.relu(-I1) + torch.relu(-I2) + torch.relu(-I3))
        order_costs = self.regular_order_cost * qr + self.expedited_order_cost * (qe1 + qe2)
        cycle_cost = order_costs + holding + shortage  # (batch,1)

        return cycle_cost  # (batch,1)

    # helpers to fetch histories for plotting / analysis (return numpy)
    def get_histories_numpy(self):
        def _stack(list_of_tensors):
            if len(list_of_tensors) == 0:
                return None
            return torch.cat(list_of_tensors, dim=1).detach().cpu().numpy()
        return {
            "I1": _stack(self._history_I1),
            "I2": _stack(self._history_I2),
            "I3": _stack(self._history_I3),
            "qe1": _stack(self._history_qe1),
            "qe2": _stack(self._history_qe2),
            "qr": _stack(self._history_qr),
            "demands": torch.cat(self._history_demands, dim=0).detach().cpu().numpy() if len(self._history_demands) else None,
            "past_inventories": self.past_inventories.detach().cpu().numpy(),
        }



class ThreeDayNeuralController(torch.nn.Module):
    """
    Simple MLP controller that takes current inventory (batch,1) and outputs
    three nonnegative values: qe1, qe2, qr. Use STE to discretize to integers if desired.
    """
    def __init__(self, hidden_sizes=[64, 32], activation=torch.nn.CELU(alpha=1.0), use_ste: bool=True):
        super().__init__()
        layers = []
        in_dim = 1  # only current inventory
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, 3))  # outputs qe1, qe2, qr
        self.net = torch.nn.Sequential(*layers)
        self.use_ste = use_ste

    def forward(self, I_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        I_t: (batch,1) tensor
        returns three tensors (batch,1): qe1, qe2, qr (already STE-floor'ed if use_ste)
        """
        raw = self.net(I_t)  # (batch,3)
        # ensure nonnegativity via e.g. softplus or relu/celu
        # here apply CELU/relu already in hidden, use softplus to guarantee positivity smoothly
        positives = torch.nn.functional.softplus(raw)  # (batch,3)
        qe1 = positives[:, [0]]
        qe2 = positives[:, [1]]
        qr  = positives[:, [2]]

        if self.use_ste:
            # convert to integer forward, keep gradients on continuous positives
            qe1 = ste_floor(qe1)
            qe2 = ste_floor(qe2)
            qr  = ste_floor(qr)
        return qe1, qe2, qr


# ---------------------------
# Training loop example
# ---------------------------
def train_three_day(
    env: ThreeDayCycleModel,
    controller: ThreeDayNeuralController,
    cycles_per_epoch: int,
    epochs: int,
    params_lr: float = 3e-3,
    init_inventory_lr: float = 1e-1,
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cpu")
    controller.to(device)
    env.init_inventory = env.init_inventory.to(device)
    env.device = device

    # Single optimizer with param groups (clean)
    optimizer = torch.optim.RMSprop(
        [
            {"params": [p for n, p in controller.named_parameters()], "lr": params_lr},
            {"params": [env.init_inventory], "lr": init_inventory_lr},
        ]
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        env.reset(batch_size=env.batch_size)  # will use env.init_inventory
        total_cycle_cost = 0.0
        for c in range(cycles_per_epoch):
            I_t = env.get_current_inventory().to(device)  # (batch,1)
            qe1, qe2, qr = controller(I_t)
            cycle_cost = env.order(qe1, qe2, qr)  # (batch,1)
            total_cycle_cost = total_cycle_cost + cycle_cost.mean()
        # average loss across cycles
        loss = total_cycle_cost / cycles_per_epoch
        loss.backward()
        # optional gradient clipping:
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs} | avg cost/cycle: {loss.item():.4f}")
    return controller, env


if __name__ == "__main__":
    ctr, env = train_three_day(env, controller, cycles_per_epoch=20, epochs=50)
    hist = env.get_histories_numpy()
    print("qe1 mean:", hist['qe1'].mean(), "qr mean:", hist['qr'].mean())
