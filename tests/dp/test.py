import logging
import random 

import matplotlib.pyplot as plt 
from tqdm import tqdm 

from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.dual_controller.dynamic_programming import DynamicProgrammingController
from src.idinn.demand import UniformDemand

logging.basicConfig(
    filename="tests/dp/test_base_dp.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_dp():

    sourcing_model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        demand_generator=UniformDemand(0,4),
        batch_size=1 
    )


    dp_controller = DynamicProgrammingController()

    dp_controller.fit(
        sourcing_model=sourcing_model,
        tolerance=0.001
    )

    random_ints = []
    for i in range(200):
        rand_int = random.randint(100, 100000)
        if rand_int not in random_ints:
            random_ints.append(rand_int)


    costs = {}

    for T in tqdm(random_ints):
        avg_cost = dp_controller.get_average_cost(
            sourcing_model=sourcing_model,
            sourcing_periods=T
        ).detach().item()

        costs[T] = avg_cost

    # --- Plot ---
    sorted_periods = sorted(costs.keys())
    sorted_costs = [costs[T] for T in sorted_periods]

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_periods, sorted_costs, marker='o', markersize=3, linewidth=1, color='steelblue')
    plt.xlabel("Time Periods (T)")
    plt.ylabel("Average Cost")
    plt.title("Average Cost vs Time Periods")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("tests/dp/avg_cost_vs_periods.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    test_dp()