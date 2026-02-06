import pandas as pd
import numpy as np

from src.idinn.sourcing_model import DualSourcingModel 
from src.idinn.periodic_controller.dynamic_programming import DynamicProgrammingController 
from src.idinn.demand import UniformDemand 
import logging 

logging.basicConfig( 
    filename="src/idinn/periodic_controller/sim_dp.log", 
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s", ) 
    
logger = logging.getLogger(__name__)

def simulate_dp(
    controller,
    model,
    T: int,
    seed: int = 42,
):
    np.random.seed(seed)

    lr = model.get_regular_lead_time()
    demand_gen = model.demand_generator

    h = model.get_holding_cost()
    b = model.get_shortage_cost()
    ce = model.get_expedited_order_cost()

    ip_e = int(model.init_inventory.item())
    pipeline = [0] * (lr - 1)
    phase = 0

    rows = []
    total_cost = 0.0

    for t in range(T):

        state = (
            int(ip_e),
            *[int(x) for x in pipeline],
            int(phase),
        )
        qr, qe = controller.qf[state]

        cost = qe * ce
        ip_e = ip_e + qe + pipeline[0]

        d = demand_gen.sample()
        ip_e -= d

        inv_on_hand = ip_e - pipeline[0]
        cost += h * inv_on_hand if inv_on_hand > 0 else b * (-inv_on_hand)

        pipeline = pipeline[1:] + [qr]
        phase = 1 - phase

        total_cost += cost

        rows.append({
            "t": int(t),
            "state": state,
            "qr": int(qr),
            "qe": int(qe),
            "demand": int(d),
            "inventory": int(ip_e),
            "period_cost": float(cost),
        })


    return pd.DataFrame(rows), total_cost / T

def test_dynamic_programming_evaluation():
    demand_ranges = [
        (0, 3),
        (0, 4),
        (2, 6),
    ]

    all_results = []

    for low, high in demand_ranges:
        demand = UniformDemand(low=low, high=high)

        model = DualSourcingModel(
            regular_lead_time=2,
            expedited_lead_time=0,
            regular_order_cost=0,
            expedited_order_cost=20,
            holding_cost=5,
            shortage_cost=495,
            init_inventory=6,
            demand_generator=demand,
            batch_size=1,
        )

        dp_controller = DynamicProgrammingController()
        dp_controller.fit(sourcing_model=model)

        sim_df, avg_sim_cost = simulate_dp(
            controller=dp_controller,
            model=model,
            T=50,
            seed=42,
        )

        dp_value = float(dp_controller.vf)
        avg_sim_cost = float(avg_sim_cost)

        dp_value = float(dp_controller.vf)
        avg_sim_cost = float(avg_sim_cost)

        sim_df["demand_low"] = int(low)
        sim_df["demand_high"] = int(high)
        sim_df["avg_sim_cost"] = avg_sim_cost
        sim_df["dp_value"] = dp_value

        all_results.append(sim_df)

        logger.info(
            f"Demand [{low},{high}] | "
            f"DP value={dp_value:.3f} | "
            f"Sim avg cost={avg_sim_cost:.3f}"
        )

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(
        "src/idinn/periodic_controller/results/sim_dp.csv", 
        index=False,
    )


if __name__ == "__main__":
    test_dynamic_programming_evaluation()