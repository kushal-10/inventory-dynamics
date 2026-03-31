import logging
import argparse
import torch
from tqdm import tqdm
import os

from src.idinn.cyclic_dual_controller.cyclic_dual_neural import CyclicDualNeuralController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="src/idinn/finetuning/finetuning_2_cycle.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Pretrained checkpoint to load weights from
PRETRAINED_PATH = "models/trained/best_model.pt"

# Where to save the fine-tuned model
FINETUNED_PATH = "models/trained/finetuned_2_cycle.pt"

EVAL_PERIODS = 1000
EVAL_SEEDS = 50


def get_finetuning_sourcing_model() -> DualSourcingModel:
    """
    New sourcing model with different costs for fine-tuning.
    Same lead times as pre-training (required for weight compatibility).
    """
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=30,    # higher expedited cost
        holding_cost=2,             # lower holding cost
        shortage_cost=198,          # lower shortage cost
        init_inventory=6,
        demand_generator=UniformDemand(0, 4),
        batch_size=1,
    )


def finetune():
    """Load pretrained best_model.pt and fine-tune on a new cost configuration."""
    if not os.path.exists(PRETRAINED_PATH):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at '{PRETRAINED_PATH}'. "
            "Run pre_training.py --train or --seed_train first."
        )

    sourcing_model = get_finetuning_sourcing_model()

    print(f"Loading pretrained weights from {PRETRAINED_PATH} ...")
    logger.info(f"Loading pretrained checkpoint from {PRETRAINED_PATH}")
    controller = CyclicDualNeuralController.load_checkpoint(
        path=PRETRAINED_PATH,
        sourcing_model=sourcing_model,
    )

    print("Fine-tuning on new sourcing model ...")
    logger.info("Starting fine-tuning")
    os.makedirs(os.path.dirname(FINETUNED_PATH), exist_ok=True)
    controller.fit(
        sourcing_model=sourcing_model,
        sourcing_periods=100,
        epochs=3000,
        validation_sourcing_periods=1000,
        parameters_lr=1e-4,         # lower LR to preserve pretrained weights
        init_inventory_lr=1e-1,
        seed=42,
        checkpoint_path=FINETUNED_PATH,
    )
    print(f"Fine-tuning complete. Checkpoint saved to {FINETUNED_PATH}")
    logger.info(f"Fine-tuning complete. Checkpoint saved to {FINETUNED_PATH}")


def infer():
    """Load the fine-tuned checkpoint and evaluate over multiple seeds."""
    if not os.path.exists(FINETUNED_PATH):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found at '{FINETUNED_PATH}'. "
            "Run with --finetune first."
        )

    sourcing_model = get_finetuning_sourcing_model()
    controller = CyclicDualNeuralController.load_checkpoint(
        path=FINETUNED_PATH,
        sourcing_model=sourcing_model,
    )

    costs = []
    with torch.no_grad():
        for seed in tqdm(range(100)):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=EVAL_PERIODS,
                seed=seed,
            )
            costs.append(cost)
            logger.info(f"Inference seed {seed} cost: {cost:.4f}")

    mean_cost = torch.mean(torch.stack(costs))
    std_cost = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost:.4f}, std: {std_cost:.4f}")
    print(f"Final mean: {mean_cost.item():.4f}")
    print(f"Final std:  {std_cost.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CyclicDualNeuralController (2-cycle) on a new cost configuration"
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=f"Load {PRETRAINED_PATH} and fine-tune on the new sourcing model",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help=f"Load {FINETUNED_PATH} and run inference",
    )
    args = parser.parse_args()

    if args.finetune:
        finetune()
    elif args.infer:
        infer()
    else:
        parser.print_help()
