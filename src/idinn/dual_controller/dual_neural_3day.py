from typing import List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from ..sourcing_model import DualSourcingModel
from .base import BaseDualController

import logging
from datetime import datetime

# Get root logger
logger = logging.getLogger()


# --- Device-safe LogReLU: convert from NumPy loop to a PyTorch Module.
# Replaces original LogReLU which used numpy / python loops (could cause CPU/GPU mismatches).
class LogReLU(torch.nn.Module):
    """
    LogReLU activation implemented in pure PyTorch so it respects device placement.
    forward: applies ReLU, then log(beta * x + 1).
    """

    def __init__(self, beta_value: float = 0.5):
        super().__init__()
        # store beta as a float (no learnable param)
        self.beta_value = float(beta_value)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # apply ReLU (keeps autograd) and then elementwise log(1 + beta * x)
        x = torch.relu(input)
        return torch.log1p(x * self.beta_value)


class DualSourcingNeuralController(torch.nn.Module, BaseDualController):
    """
    DualSourcingNeuralController with autoregressive unrolling:
    - Within each period, we unroll 3 sub-days.
      * day 1: expedited only
      * day 2: expedited only
      * day 3: regular only
    - Each sub-day we:
      1. Prepare inputs from the current simulated state
      2. Predict order for that sub-day
      3. Apply order via sourcing_model.order(...) (this updates inventory and samples demand)
      4. Accumulate immediate cost (ordering + holding/shortage) for that sub-day
    - The unrolled costs across the 3 sub-days are summed and used as the differentiable loss.
    """

    def __init__(
            self,
            hidden_layers: List[int] = [128, 64, 32, 16, 8, 4],
            activation: torch.nn.Module = torch.nn.ReLU(),  # Activation for hidden layers
            compressed: bool = False,
            regular_activation: [torch.nn.Module | str] = torch.nn.ReLU(),  # Activation for day3 (regular)
            expedited_activation: torch.nn.Module = torch.nn.ReLU(),  # Activation for days1-2 (expedited)
    ):
        super().__init__()
        self.sourcing_model = None
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.compressed = compressed

        # Will be used for the hidden layers sequence
        self.hidden_sequence = None
        # Final linear head that produces 3 pre-activations (d1, d2, d3)
        self.final_linear = None

        # keep track of unique integer orders seen (diagnostics)
        self.expedited_q_set = set()
        self.regular_q_set = set()

        # store activations (these can be nn.Module or numeric string for LogReLU)
        self.regular_activation = regular_activation
        self.expedited_activation = expedited_activation
        logger.info(
            f"Initialized DualSourcingNeuralController with hidden_layers={hidden_layers}, compressed={compressed}")

    def init_layers(self, regular_lead_time: int, expedited_lead_time: int) -> None:
        """
        Initialize the shared encoder (hidden_sequence) and the final_linear head.

        NOTE: unchanged semantics from original code, except the final head
        is interpreted as three outputs: day1_exp, day2_exp, day3_reg.
        """
        if self.compressed:
            input_length = regular_lead_time + expedited_lead_time
        else:
            input_length = regular_lead_time + expedited_lead_time + 1

        # Build the MLP body (same as before)
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

        self.hidden_sequence = torch.nn.Sequential(*architecture)

        # Final linear layer now outputs 3 activations: day1_preact, day2_preact, day3_preact
        # (we keep shape (hidden_last -> 3) consistent with previous "3 output" comment)
        self.final_linear = torch.nn.Linear(self.hidden_layers[-1], 3)
        logger.info(f"Initialized NN with three output neurons (day1_exp, day2_exp, day3_reg), "
                    f"regular lead time: {regular_lead_time}, expedited lead time: {expedited_lead_time}")

    def prepare_inputs(
            self,
            current_inventory: torch.Tensor,
            past_regular_orders: torch.Tensor,
            past_expedited_orders: torch.Tensor,
            sourcing_model: DualSourcingModel,
    ) -> torch.Tensor:
        """
        Same as original prepare_inputs: checks shapes / pads pipelines and concatenates
        current_inventory + last regular pipeline + last expedited pipeline.
        """
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
        return inputs

    # NOTE: we do not change the signature of forward(inputs). It still accepts a
    # pre-built inputs tensor and returns three floored order tensors (reg/exp mapping below).
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that outputs three orders (as *floored* integer tensors using STE).
        Returns: (exp_d1, exp_d2, reg_d3) each shaped (batch,1)
        (day1 and day2 are expedited; day3 is regular).
        """
        # 1. Body
        h_hidden = self.hidden_sequence(inputs)

        # 2. Final linear producing 3 pre-activations
        h = self.final_linear(h_hidden)  # [batch_size, 3] -> (day1_preact, day2_preact, day3_preact)

        # 3. Split preactivations
        h_d1 = h[:, 0].unsqueeze(1)  # day1 preactivation (expedited)
        h_d2 = h[:, 1].unsqueeze(1)  # day2 preactivation (expedited)
        h_d3 = h[:, 2].unsqueeze(1)  # day3 preactivation (regular)

        # 4. Apply activation functions (expedited for d1,d2 / regular for d3)
        # Support both nn.Module and string-specified LogReLU (float-as-string)
        if type(self.expedited_activation) == str:
            beta_value = float(self.expedited_activation)
            expedited_act = LogReLU(beta_value=beta_value).to(h_d1.device)
            expedited_h1 = expedited_act(h_d1)
            expedited_h2 = expedited_act(h_d2)
        else:
            expedited_h1 = self.expedited_activation(h_d1)
            expedited_h2 = self.expedited_activation(h_d2)

        if type(self.regular_activation) == str:
            beta_value = float(self.regular_activation)
            regular_act = LogReLU(beta_value=beta_value).to(h_d3.device)
            regular_h = regular_act(h_d3)
        else:
            regular_h = self.regular_activation(h_d3)

        # cast to float explicitly (safety)
        expedited_h1 = expedited_h1.float()
        expedited_h2 = expedited_h2.float()
        regular_h = regular_h.float()

        # 5. Fractional decoupling (straight-through):
        # q = h - frac(h).detach()  => forward uses floor(h) while gradients pass to h.
        exp_q1 = expedited_h1 - torch.frac(expedited_h1).clone().detach()
        exp_q2 = expedited_h2 - torch.frac(expedited_h2).clone().detach()
        reg_q = regular_h - torch.frac(regular_h).clone().detach()

        exp_d1 = exp_q1  # day1 expedited order
        exp_d2 = exp_q2  # day2 expedited order
        reg_d3 = reg_q    # day3 regular order

        # convert to same dtype
        exp_d1 = exp_d1.float()
        exp_d2 = exp_d2.float()
        reg_d3 = reg_d3.float()

        return exp_d1, exp_d2, reg_d3

    def predict_step(
            self,
            current_inventory: Union[int, torch.Tensor],
            past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
            past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
            step: int = 1,
            output_tensor: bool = False,
    ) -> Union[torch.Tensor, int]:
        """
        Predict a single sub-day order for the specified step (1,2,3).
        This is used to autoregressively unroll the 3 sub-days.

        Parameters:
         - step: 1 or 2 -> returns expedited order for that day
                 3 -> returns regular order for that day
        Returns:
         - torch.Tensor ((batch,1)) if output_tensor True, else int scalar (for batch_size=1)
        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained or sourcing model not set.")

        # prepare inputs based on the current state (same as in predict)
        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model
        )
        # get the three day outputs (we only use one depending on step)
        exp_d1, exp_d2, reg_d3 = self.forward(inputs)

        if step == 1:
            out = exp_d1
        elif step == 2:
            out = exp_d2
        elif step == 3:
            out = reg_d3
        else:
            raise ValueError("step must be 1, 2, or 3")

        if output_tensor:
            return out
        else:
            # Convert to python int when batch_size=1 for compatibility
            return int(out.item())

    # --- NEW METHOD (major change) ---
    def get_total_cost(
            self,
            sourcing_model: DualSourcingModel,
            sourcing_periods: int,
            seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Controller-local simulator that unrolls 3 sub-days per period (autoregressive).
        It returns the total cost (torch scalar) accumulated over `sourcing_periods`.
        This replaces using the base-class get_total_cost during training.

        Implementation details:
         - For each of `sourcing_periods`:
            * we sequentially predict day1 (exp), apply it -> sourcing_model.order(regular_q=0, expedited_q=exp1)
              (this samples demand and updates inventory inside sourcing_model)
            * we compute immediate costs for that sub-day (ordering + holding/shortage)
            * repeat for day2 (exp) and day3 (reg)
         - All operations use tensors so autograd tracks gradients through the controller outputs
           (note: demand draws are nondifferentiable; the rest is differentiable).
         - We return the total (sum) cost as a single torch scalar (float).
        """

        # store sourcing_model in controller for convenience (used by predict_step)
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        device = sourcing_model.get_init_inventory().device  # ensure device alignment

        # cost accumulators
        total_cost = torch.tensor(0.0, device=device, dtype=torch.float)

        # local copies for readability
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        cr = sourcing_model.get_regular_order_cost()

        batch_size = sourcing_model.batch_size

        # We'll run sourcing_periods episodes; each episode contains 3 sub-day updates
        for period in range(sourcing_periods):
            # For reproducibility, optionally reseed here (the original code reseeded in order method)
            # but we rely on the `seed` above and torch's RNG.

            # Sub-day 1: expedited only
            # Prepare inputs (current state)
            cur_inv = sourcing_model.get_current_inventory()  # shape (batch,1)
            past_reg = sourcing_model.get_past_regular_orders() if hasattr(sourcing_model, "get_past_regular_orders") else torch.zeros(batch_size, 1, device=device)
            past_exp = sourcing_model.get_past_expedited_orders() if hasattr(sourcing_model, "get_past_expedited_orders") else torch.zeros(batch_size, 1, device=device)

            # Predict expedited order for day1 (tensor)
            exp_d1 = self.predict_step(cur_inv, past_reg, past_exp, step=1, output_tensor=True)
            # Ensure exp_d1 has right shape / device
            exp_d1 = exp_d1.to(device).float()

            # Apply the order: regular_q=0 for this sub-day; expedited order placed in expedited pipeline
            sourcing_model.order(regular_q=torch.zeros_like(exp_d1), expedited_q=exp_d1)

            # Compute immediate ordering cost for sub-day 1
            # Note: regular order zero so cost = ce * exp_d1
            order_cost_1 = (ce * exp_d1).sum()  # sum across batch if batched
            # inventory after demand is inside sourcing_model.get_current_inventory()
            inv_after_1 = sourcing_model.get_current_inventory()
            # holding/shortage cost (elementwise)
            holding_1 = torch.clamp(inv_after_1, min=0.0).sum() * h
            shortage_1 = -torch.clamp(inv_after_1, max=0.0).sum() * b
            total_cost = total_cost + order_cost_1 + holding_1 + shortage_1

            # Sub-day 2: expedited only (autoregressive: the predict sees updated state)
            cur_inv = sourcing_model.get_current_inventory()
            past_reg = sourcing_model.get_past_regular_orders()
            past_exp = sourcing_model.get_past_expedited_orders()

            exp_d2 = self.predict_step(cur_inv, past_reg, past_exp, step=2, output_tensor=True)
            exp_d2 = exp_d2.to(device).float()

            sourcing_model.order(regular_q=torch.zeros_like(exp_d2), expedited_q=exp_d2)

            order_cost_2 = (ce * exp_d2).sum()
            inv_after_2 = sourcing_model.get_current_inventory()
            holding_2 = torch.clamp(inv_after_2, min=0.0).sum() * h
            shortage_2 = -torch.clamp(inv_after_2, max=0.0).sum() * b
            total_cost = total_cost + order_cost_2 + holding_2 + shortage_2

            # Sub-day 3: regular only
            cur_inv = sourcing_model.get_current_inventory()
            past_reg = sourcing_model.get_past_regular_orders()
            past_exp = sourcing_model.get_past_expedited_orders()

            reg_d3 = self.predict_step(cur_inv, past_reg, past_exp, step=3, output_tensor=True)
            reg_d3 = reg_d3.to(device).float()

            # Apply the regular order now (expedited = 0)
            sourcing_model.order(regular_q=reg_d3, expedited_q=torch.zeros_like(reg_d3))

            order_cost_3 = (cr * reg_d3).sum()
            inv_after_3 = sourcing_model.get_current_inventory()
            holding_3 = torch.clamp(inv_after_3, min=0.0).sum() * h
            shortage_3 = -torch.clamp(inv_after_3, max=0.0).sum() * b
            total_cost = total_cost + order_cost_3 + holding_3 + shortage_3

            # Optionally, you may want to aggregate per-period averages, but we sum here
            # to match original training (paper uses total or average).

        # return total cost (torch scalar)
        return total_cost

    def predict(
            self,
            current_inventory: Union[int, torch.Tensor],
            past_regular_orders: Optional[Union[List[int], torch.Tensor]] = None,
            past_expedited_orders: Optional[Union[List[int], torch.Tensor]] = None,
            output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """
        Backwards-compatible predict method: for compatibility with code expecting a single
        (regular, expedited) pair, we will:
         - run the autoregressive sequence on the *current state* (without changing sourcing_model),
           but only here we will NOT call sourcing_model.order (so this is purely inference).
         - return a tuple containing (regular_q_day3, expedited_q_day2) as an example,
           or return tensors if output_tensor True.

        Note: this function is primarily kept for compatibility. You should prefer
        predict_step during unrolled training/inference for correct sequencing.
        """
        # Make sure controller has a sourcing_model to know lead times etc.
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        # Prepare inputs and do a forward pass to get the three day outputs
        inputs = self.prepare_inputs(
            current_inventory,
            past_regular_orders,
            past_expedited_orders,
            sourcing_model=self.sourcing_model
        )
        exp_d1, exp_d2, reg_d3 = self.forward(inputs)

        # For compatibility, return reg_d3 (regular) and exp_d2 (expedited)
        if output_tensor:
            return reg_d3, exp_d2
        else:
            return int(reg_d3.item()), int(exp_d2.item())

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
            device: Optional[torch.device] = None,
    ):
        """
        Fit method modified to call controller-local get_total_cost(...) that performs
        autoregressive unrolling for each period (3 sub-days).
        Key change: replaced call to super().get_total_cost with self.get_total_cost.
        Everything else is kept similar to original for training/optimizers/logging.
        """
        # Store sourcing model for predict_step usage
        self.sourcing_model = sourcing_model

        if seed is not None:
            torch.manual_seed(seed)

        if self.final_linear is None:
            self.init_layers(
                regular_lead_time=sourcing_model.get_regular_lead_time(),
                expedited_lead_time=sourcing_model.get_expedited_lead_time(),
            )
        if device is not None:
            # Move controller to device
            self.to(device)
            sourcing_model.init_inventory = (
                sourcing_model.init_inventory.detach().to(device).clone().requires_grad_(True)
            )

        start_time = datetime.now()
        logger.info(f"Starting dual sourcing neural network training at {start_time}")
        logger.info(f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
                    f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
                    f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}")
        logger.info(f"Training parameters: epochs={epochs}, sourcing_periods={sourcing_periods}, "
                    f"validation_periods={validation_sourcing_periods}, learning_rate={parameters_lr}")

        # two optimizers: one for init inventory and one for NN parameters
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
            # ===== MAJOR CHANGE: call controller-local get_total_cost that unrolls 3 sub-days ====
            train_loss = self.get_total_cost(sourcing_model, sourcing_periods)
            # ===================================================================================
            train_loss.backward()
            # Perform gradient descend
            if epoch % init_inventory_freq == 0:
                optimizer_init_inventory.step()
            else:
                optimizer_parameters.step()
            # Save the best model
            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                eval_loss = self.get_total_cost(
                    sourcing_model, validation_sourcing_periods, seed=seed
                )
                if eval_loss < min_loss:
                    min_loss = eval_loss
                    best_state = self.state_dict()
            else:
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_state = self.state_dict()
            # Log train loss
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "Avg. cost per period/train", train_loss / sourcing_periods, epoch
                )
                if validation_sourcing_periods is not None and epoch % 10 == 0:
                    # Log validation loss
                    tensorboard_writer.add_scalar(
                        "Avg. cost per period/val",
                        eval_loss / validation_sourcing_periods,
                        epoch,
                    )
                tensorboard_writer.flush()

            end_time = datetime.now()
            duration = end_time - start_time
            per_epoch_time = duration.total_seconds() / (epoch + 1)  # seconds per epoch
            remaining_time = (epochs - epoch) * per_epoch_time
            if epoch % log_freq == 0:
                logger.info(f"Epoch {epoch}/{epochs}"
                            f" - Training cost: {train_loss / sourcing_periods:.4f}"
                            f" - Per epoch time: {per_epoch_time:.2f} seconds"
                            f" - Est. Remaining time: {int(remaining_time)} seconds."
                            )

            if validation_sourcing_periods is not None and epoch % validation_freq == 0:
                logger.info(f"Epoch {epoch}/{epochs}"
                            f" - Validation cost: {eval_loss / validation_sourcing_periods:.4f}"
                            f" - Per epoch time: {per_epoch_time:.2f} seconds"
                            f" - Est. Remaining time: {int(remaining_time)} seconds."
                            )

        # load and return best state
        self.load_state_dict(best_state)

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Training completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(f"Final best cost: {min_loss / sourcing_periods:.4f}")

        return self.regular_q_set, self.expedited_q_set


    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None and self.batch_size != batch_size:
            self.batch_size = batch_size

        # Ensure all tensors created here use the same device/dtype as init_inventory
        init_inv = self.get_init_inventory()
        device = init_inv.device
        dtype = init_inv.dtype

        # past_inventories: repeat init inventory on correct device
        self.past_inventories = init_inv.repeat(self.batch_size, 1).to(device=device, dtype=dtype)

        # past_demands: zeros on same device/dtype
        self.past_demands = torch.zeros(self.batch_size, 1, device=device, dtype=dtype)

        if self.lead_time is not None:
            # single sourcing pipeline
            self.past_orders = torch.zeros(self.batch_size, 1, device=device, dtype=dtype)

        elif (
            self.regular_lead_time is not None and self.expedited_lead_time is not None
        ):
            if self.regular_lead_time < self.expedited_lead_time:
                raise ValueError(
                    "`regular_lead_time` must be greater than or equal to expedited_lead_time."
                )
            # dual sourcing pipelines
            self.past_regular_orders = torch.zeros(self.batch_size, 1, device=device, dtype=dtype)
            self.past_expedited_orders = torch.zeros(self.batch_size, 1, device=device, dtype=dtype)
        else:
            raise ValueError(
                "Either `lead_time` or (`regular_lead_time` and `expedited_lead_time`) must be provided."
            )


    def save(self, path: str) -> None:
        # Save the PyTorch module state (parameters)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        # Load a state dict into the model (assumes same architecture)
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
