"""
Run simulations for DP and NN phase controllers on two sourcing models:
  - Backorder cost b=95  (NN checkpoint: models/trained/finetuned_b95.pt)
  - Backorder cost b=495 (NN checkpoint: models/trained/best_model.pt)

Each simulation runs a single demand sequence of 100 periods (50 sourcing cycles).
The same 100 demand values are shared across all 4 models via a fixed random seed.
Results are saved as CSVs under results/simulation/.

Usage:
    python run_simulations.py
"""

import logging

from src.idinn.demand import UniformDemand
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.phase_controller.dp.dynamic_programming import DynamicProgrammingController
from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
from src.idinn.phase_controller.simulations import run_simulation

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
N_SEEDS = 1           # single shared demand sequence across all models
SOURCING_PERIODS = 50  # 2 actual periods per cycle → 100 periods total
DP_TOLERANCE = 10e-3
OUTPUT_DIR = "results/simulation"

# Shared sourcing model parameters (both models differ only in shortage_cost)
BASE_PARAMS = dict(
    regular_lead_time=3,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    demand_generator=UniformDemand(0, 4),
    batch_size=1,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_dp_sourcing_model(shortage_cost: float) -> DualSourcingModel:
    """Sourcing model as defined in test_cycle_dp (init_inventory=0)."""
    return DualSourcingModel(**BASE_PARAMS, shortage_cost=shortage_cost, init_inventory=0)


def make_nn_sourcing_model(shortage_cost: float) -> DualSourcingModel:
    """
    Sourcing model for NN inference (init_inventory=6 as training baseline;
    will be overridden by the value stored in the checkpoint).
    """
    return DualSourcingModel(**BASE_PARAMS, shortage_cost=shortage_cost, init_inventory=6)


# ---------------------------------------------------------------------------
# DP simulations
# ---------------------------------------------------------------------------

def run_dp_simulation(shortage_cost: float, output_path: str) -> None:
    print(f"\n=== DP  |  b={shortage_cost} ===")
    sourcing_model = make_dp_sourcing_model(shortage_cost)

    print("  Fitting DP controller (this may take a few minutes)...")
    controller = DynamicProgrammingController()
    controller.fit(sourcing_model=sourcing_model, tolerance=DP_TOLERANCE)

    print(f"  Running simulation ({SOURCING_PERIODS} cycles, 100 shared demand periods)...")
    run_simulation(
        controller=controller,
        sourcing_model=sourcing_model,
        n_seeds=N_SEEDS,
        sourcing_periods=SOURCING_PERIODS,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# NN simulations
# ---------------------------------------------------------------------------

def run_nn_simulation(shortage_cost: float, checkpoint_path: str, output_path: str) -> None:
    print(f"\n=== NN  |  b={shortage_cost}  |  checkpoint: {checkpoint_path} ===")
    sourcing_model = make_nn_sourcing_model(shortage_cost)

    print("  Loading NN checkpoint...")
    controller = MultiPeriodNeuralController.load_checkpoint(
        path=checkpoint_path,
        sourcing_model=sourcing_model,
    )
    controller.eval()

    print(f"  Running simulation ({SOURCING_PERIODS} cycles, 100 shared demand periods)...")
    run_simulation(
        controller=controller,
        sourcing_model=sourcing_model,
        n_seeds=N_SEEDS,
        sourcing_periods=SOURCING_PERIODS,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # DP — b=95
    # run_dp_simulation(
    #     shortage_cost=95,
    #     output_path=f"{OUTPUT_DIR}/dp_b95.csv",
    # )

    # DP — b=495
    run_dp_simulation(
        shortage_cost=495,
        output_path=f"{OUTPUT_DIR}/dp_b495.csv",
    )

    # # NN — b=95
    # run_nn_simulation(
    #     shortage_cost=95,
    #     checkpoint_path="models/trained/finetuned_b95.pt",
    #     output_path=f"{OUTPUT_DIR}/nn_b95.csv",
    # )

    # # NN — b=495
    # run_nn_simulation(
    #     shortage_cost=495,
    #     checkpoint_path="models/trained/best_model.pt",
    #     output_path=f"{OUTPUT_DIR}/nn_b495.csv",
    # )

    print("\nAll simulations complete. Results saved to:", OUTPUT_DIR)
