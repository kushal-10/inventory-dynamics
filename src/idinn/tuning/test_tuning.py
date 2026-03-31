"""
Standalone train + eval script for CyclicDualSourcingNeuralController.

Runs a single training run with fixed hyperparameters, logs every epoch,
and prints gradient/parameter norms to help diagnose NaN issues.

Usage:
    uv run python src/idinn/tuning/test_tuning.py
"""

import logging
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Logging — stdout + file
# ---------------------------------------------------------------------------
_LOG_FILE = Path(__file__).parent / "test_tuning.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — edit these to test different settings
# ---------------------------------------------------------------------------
CYCLE_LENGTH = 2
HIDDEN_LAYERS = [64, 32, 16]
ACTIVATION = torch.nn.ReLU()
COMPRESSED = False

EPOCHS = 500
SOURCING_PERIODS = 50
BATCH_SIZE = 8
PARAMETERS_LR = 1e-3
INIT_INVENTORY_LR = 1e-2
INIT_INVENTORY_FREQ = 4
SEED = 42

REGULAR_LEAD_TIME = 2
EXPEDITED_LEAD_TIME = 0
REGULAR_ORDER_COST = 0
EXPEDITED_ORDER_COST = 20
HOLDING_COST = 5
SHORTAGE_COST = 495
INIT_INVENTORY = 0
DEMAND_LOW = 0
DEMAND_HIGH = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sourcing_model(batch_size: int):
    from src.idinn.demand import UniformDemand
    from src.idinn.sourcing_model import DualSourcingModel

    return DualSourcingModel(
        regular_lead_time=REGULAR_LEAD_TIME,
        expedited_lead_time=EXPEDITED_LEAD_TIME,
        regular_order_cost=REGULAR_ORDER_COST,
        expedited_order_cost=EXPEDITED_ORDER_COST,
        holding_cost=HOLDING_COST,
        shortage_cost=SHORTAGE_COST,
        init_inventory=INIT_INVENTORY,
        demand_generator=UniformDemand(low=DEMAND_LOW, high=DEMAND_HIGH),
        batch_size=batch_size,
    )


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


def param_norm(params) -> float:
    total = 0.0
    for p in params:
        total += p.detach().norm().item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)

    from src.idinn.cyclic_dual_controller.cyclic_dual_neural import (
        CyclicDualSourcingNeuralController,
    )

    sourcing_model = make_sourcing_model(batch_size=BATCH_SIZE)

    controller = CyclicDualSourcingNeuralController(
        cycle_length=CYCLE_LENGTH,
        hidden_layers=HIDDEN_LAYERS,
        activation=ACTIVATION,
        compressed=COMPRESSED,
    )
    controller.init_layers(
        regular_lead_time=REGULAR_LEAD_TIME,
        expedited_lead_time=EXPEDITED_LEAD_TIME,
    )
    controller.sourcing_model = sourcing_model

    optimizer_inv = torch.optim.RMSprop(
        [sourcing_model.init_inventory], lr=INIT_INVENTORY_LR
    )
    optimizer_params = torch.optim.RMSprop(controller.parameters(), lr=PARAMETERS_LR)

    logger.info("=" * 60)
    logger.info(f"cycle_length={CYCLE_LENGTH}  hidden={HIDDEN_LAYERS}  activation={ACTIVATION}")
    logger.info(f"parameters_lr={PARAMETERS_LR}  init_inventory_lr={INIT_INVENTORY_LR}  init_inventory_freq={INIT_INVENTORY_FREQ}")
    logger.info(f"epochs={EPOCHS}  sourcing_periods={SOURCING_PERIODS}  batch={BATCH_SIZE}")
    logger.info("=" * 60)

    for epoch in range(EPOCHS):
        optimizer_inv.zero_grad()
        optimizer_params.zero_grad()
        sourcing_model.reset()

        loss = controller.get_total_cost(sourcing_model, SOURCING_PERIODS)

        loss_val = loss.item()
        init_inv_val = sourcing_model.init_inventory.item()

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"Epoch {epoch:3d} | loss={loss_val} | init_inv={init_inv_val:.4f} | "
                f"param_norm={param_norm(controller.parameters()):.4f} | STOPPING"
            )
            break

        loss.backward()

        nn_grad = grad_norm(controller.parameters())
        inv_grad = (
            sourcing_model.init_inventory.grad.item()
            if sourcing_model.init_inventory.grad is not None else 0.0
        )

        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([sourcing_model.init_inventory], max_norm=1.0)

        nn_grad_clipped = grad_norm(controller.parameters())

        if epoch % INIT_INVENTORY_FREQ == 0:
            optimizer_inv.step()
        else:
            optimizer_params.step()

        avg_cost = loss_val / (SOURCING_PERIODS * CYCLE_LENGTH)
        logger.info(
            f"Epoch {epoch:3d} | avg_cost={avg_cost:10.4f} | "
            f"init_inv={init_inv_val:7.3f} | "
            f"nn_grad={nn_grad:8.4f} -> clipped={nn_grad_clipped:.4f} | "
            f"inv_grad={inv_grad:10.4f}"
        )

    # --- final eval ---
    logger.info("=" * 60)
    logger.info("Evaluating...")
    eval_model = make_sourcing_model(batch_size=1)
    avg_cost = controller.get_average_cost(eval_model, sourcing_periods=500, seed=0)
    logger.info(f"Final avg cost/period: {avg_cost.item():.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
