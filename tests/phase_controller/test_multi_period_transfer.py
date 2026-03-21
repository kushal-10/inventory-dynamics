"""
Fine-tune the best multi-period neural controller on a sourcing model
with shortage_cost=95 (instead of the original 495).

Usage:
    python finetune_backorder_95.py [--epochs 5000] [--seed_finetune] [--infer]

The script loads models/trained/best_model.pt as a warm start and continues
training with the updated cost parameters.
"""

import argparse
import logging
import os
import shutil
from typing import Optional

import torch
from tqdm import tqdm

from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRETRAINED_PATH    = "models/trained/best_model.pt"          # warm-start source
FINETUNED_PATH     = "models/trained/finetuned.pt"       # single fine-tune output
SEEDED_DIR         = "models/trained/seeded_b95"             # per-seed checkpoints
BEST_FINETUNED_PATH = "models/trained/best_finetuned.pt" # best across seeds

# ---------------------------------------------------------------------------
# Eval / seed settings (keep consistent with original script)
# ---------------------------------------------------------------------------
N_SEEDS       = 10
EVAL_PERIODS  = 1000
EVAL_SEEDS    = 50

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="tests/phase_controller/finetune_backorder_95.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory — only shortage_cost changed
# ---------------------------------------------------------------------------
def get_sourcing_model() -> DualSourcingModel:
    """Same as original except shortage_cost=95."""
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=95,          # <-- changed from 495
        init_inventory=6,
        demand_generator=UniformDemand(0, 4),
        batch_size=1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_warm_start(sourcing_model: DualSourcingModel) -> MultiPeriodNeuralController:
    """Load the pre-trained checkpoint as a warm start."""
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(
            f"Pre-trained checkpoint not found at '{PRETRAINED_PATH}'. "
            "Run the original training script first."
        )
    controller = MultiPeriodNeuralController.load_checkpoint(
        path=PRETRAINED_PATH,
        sourcing_model=sourcing_model,
    )
    logger.info(f"Loaded warm-start weights from {PRETRAINED_PATH}")
    return controller


def _evaluate(controller: MultiPeriodNeuralController,
               sourcing_model: DualSourcingModel) -> tuple[float, float]:
    """Return (mean_cost, std_cost) averaged over EVAL_SEEDS × EVAL_PERIODS."""
    costs = []
    with torch.no_grad():
        for eval_seed in range(EVAL_SEEDS):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=EVAL_PERIODS,
                seed=eval_seed,
            )
            costs.append(cost)
    stacked = torch.stack(costs)
    return torch.mean(stacked).item(), torch.std(stacked).item()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def finetune(epochs: int = 5000, lr: float = 1e-4):
    """
    Fine-tune a single model loaded from PRETRAINED_PATH and save to
    FINETUNED_PATH.

    A lower LR (default 1e-4 vs the original 3e-4) is recommended because
    the weights are already close to a good solution.
    """
    if os.path.exists(FINETUNED_PATH):
        raise RuntimeError(
            f"Output path '{FINETUNED_PATH}' already exists. "
            "Move or delete it before re-running."
        )

    sourcing_model = get_sourcing_model()
    controller = _load_warm_start(sourcing_model)

    print(f"Fine-tuning for {epochs} epochs with lr={lr} on shortage_cost=95 …")
    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=epochs,
        validation_sourcing_periods=1000,
        parameters_lr=lr,
        seed=42,
        checkpoint_path=FINETUNED_PATH,
    )
    print(f"Fine-tuning complete. Checkpoint saved to '{FINETUNED_PATH}'")
    logger.info(f"Single fine-tune done → {FINETUNED_PATH}")


def seed_finetune(epochs: int = 5000, lr: float = 1e-4):
    """
    Fine-tune from PRETRAINED_PATH across N_SEEDS random seeds and keep the
    best-performing model in BEST_FINETUNED_PATH.
    """
    os.makedirs(SEEDED_DIR, exist_ok=True)

    best_cost = float("inf")
    best_seed = None

    for i in range(1, N_SEEDS + 1):
        seed = i - 1  # seeds 0..9
        checkpoint_path = os.path.join(SEEDED_DIR, f"model{i}.pt")

        if os.path.exists(checkpoint_path):
            logger.info(f"Seed {seed} — checkpoint already exists, skipping training.")
            print(f"[Seed {seed}] Checkpoint exists, skipping training.")
        else:
            print(f"\n[Seed {seed}] Fine-tuning model {i}/{N_SEEDS} …")
            logger.info(f"Seed {seed} — starting fine-tune, saving to {checkpoint_path}")

            sourcing_model = get_sourcing_model()
            controller = _load_warm_start(sourcing_model)
            controller.fit(
                sourcing_model=sourcing_model,
                sourcing_periods=100,
                epochs=epochs,
                validation_sourcing_periods=1000,
                parameters_lr=lr,
                seed=seed,
                checkpoint_path=checkpoint_path,
            )

        # Evaluate
        print(f"[Seed {seed}] Evaluating …")
        sourcing_model = get_sourcing_model()
        controller = MultiPeriodNeuralController.load_checkpoint(
            path=checkpoint_path,
            sourcing_model=sourcing_model,
        )
        mean_cost, std_cost = _evaluate(controller, sourcing_model)

        logger.info(f"Seed {seed} — mean cost: {mean_cost:.4f}, std: {std_cost:.4f}")
        print(f"[Seed {seed}] Mean: {mean_cost:.4f}, Std: {std_cost:.4f}")

        if mean_cost < best_cost:
            best_cost = mean_cost
            best_seed = seed
            shutil.copy(checkpoint_path, BEST_FINETUNED_PATH)
            logger.info(
                f"New best — seed {seed}, mean cost {mean_cost:.4f}, "
                f"saved to {BEST_FINETUNED_PATH}"
            )
            print(f"[Seed {seed}] New best! Saved to '{BEST_FINETUNED_PATH}'")

    print(f"\nSeed fine-tuning complete.")
    print(f"Best seed: {best_seed}, Best mean cost: {best_cost:.4f}")
    print(f"Best model saved to '{BEST_FINETUNED_PATH}'")
    logger.info(
        f"Seed fine-tune done. Best seed: {best_seed}, cost: {best_cost:.4f}"
    )


def infer(checkpoint: Optional[str] = None):
    """
    Load a fine-tuned checkpoint (defaults to FINETUNED_PATH) and run
    inference over 500 seeds × 1000 periods.
    """
    path = checkpoint or FINETUNED_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at '{path}'.")

    sourcing_model = get_sourcing_model()
    controller = MultiPeriodNeuralController.load_checkpoint(
        path=path,
        sourcing_model=sourcing_model,
    )

    costs = []
    with torch.no_grad():
        for seed in tqdm(range(50), desc="Inference"):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=10000,
                seed=seed,
            )
            costs.append(cost)
            logger.info(f"Seed {seed} cost: {cost:.4f}")

    stacked     = torch.stack(costs)
    mean_cost   = torch.mean(stacked).item()
    std_cost    = torch.std(stacked).item()

    logger.info(f"Inference complete — mean: {mean_cost:.4f}, std: {std_cost:.4f}")
    print(f"\nFinal mean cost : {mean_cost:.4f}")
    print(f"Final std cost  : {std_cost:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the best multi-period controller on shortage_cost=95"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune a single model from the pre-trained checkpoint",
    )
    parser.add_argument(
        "--seed_finetune",
        action="store_true",
        help="Fine-tune across multiple seeds, keep the best model",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Load a fine-tuned checkpoint and run inference",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Number of fine-tuning epochs (default: 5000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (default: 1e-4, lower than original 3e-4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for --infer (default: models/trained/finetuned_b95.pt)",
    )
    args = parser.parse_args()

    if args.finetune:
        finetune(epochs=args.epochs, lr=args.lr)
    elif args.seed_finetune:
        seed_finetune(epochs=args.epochs, lr=args.lr)
    elif args.infer:
        infer(checkpoint=args.checkpoint)
    else:
        parser.print_help()