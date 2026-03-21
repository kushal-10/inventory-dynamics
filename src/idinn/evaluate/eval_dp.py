import json
import logging
import os
import time

from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.phase_controller.dp.dynamic_programming import DynamicProgrammingController
from src.idinn.demand import UniformDemand

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EVAL_DIR, "sourcing_config.json")

TOLERANCE_BY_LR = {
    2: 10e-6,
    3: 10e-5,
    4: 10e-4,
}


def load_configs():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_configs(configs):
    with open(CONFIG_PATH, "w") as f:
        json.dump(configs, f, indent=4)


def evaluate_dp(config: dict) -> dict:
    """Run DP for a single config entry. Returns updated config with 'cost' and 'time'."""
    lr       = config["lr"]
    ce       = config["ce"]
    b        = config["b"]
    demand   = config["demand"]
    tol      = TOLERANCE_BY_LR[lr]

    sourcing_model = DualSourcingModel(
        regular_lead_time=lr,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=ce,
        holding_cost=5,
        shortage_cost=b,
        init_inventory=0,
        demand_generator=UniformDemand(0, demand),
        batch_size=1,
    )

    dp_controller = DynamicProgrammingController()

    start = time.perf_counter()
    dp_controller.fit(sourcing_model=sourcing_model, tolerance=tol)
    elapsed = time.perf_counter() - start

    cost = dp_controller.cost_per_cycle   # adjust attribute name if needed

    updated = {**config, "cost": round(cost, 6), "time": round(elapsed, 4)}
    logger.info(
        f"lr={lr} ce={ce} b={b} demand={demand} | "
        f"tol={tol:.0e} | cost={cost:.4f} | time={elapsed:.2f}s"
    )
    return updated


def main():
    configs = load_configs()

    pending = [i for i, c in enumerate(configs) if "time" not in c or "cost" not in c]
    total   = len(configs)
    remaining = len(pending)

    logger.info(f"Total configs: {total} | Already done: {total - remaining} | Pending: {remaining}")

    try:
        for idx in pending:
            cfg = configs[idx]
            logger.info(
                f"Running [{idx + 1}/{total}] "
                f"lr={cfg['lr']} ce={cfg['ce']} b={cfg['b']} demand={cfg['demand']}"
            )
            configs[idx] = evaluate_dp(cfg)
            save_configs(configs)   # persist after every entry so Ctrl+C is safe

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — progress saved. Re-run to resume.")

    done = sum(1 for c in configs if "time" in c and "cost" in c)
    logger.info(f"Finished {done}/{total} configs.")


if __name__ == "__main__":
    main()