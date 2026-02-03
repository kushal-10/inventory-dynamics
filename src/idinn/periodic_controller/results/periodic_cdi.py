from datetime import datetime, timedelta
import logging

from tqdm import tqdm

from src.idinn.periodic_controller.periodic_cdi_controller import PeriodicCDIController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

logging.basicConfig(
    filename="src/idinn/periodic_controller/results/periodic_cdi.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

def calculate_periodic_cdi_avg_cost():

    model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=UniformDemand(0,4),
        batch_size=1,
    )

    cyc_cdi_controller = PeriodicCDIController()

    total_avg_cost = 0
    total_fit_duration = timedelta(0)
    total_eval_duration = timedelta(0)
    realizations = 20

    min_cost = 100000
    min_cost_seed = -1

    for seed in tqdm(range(realizations)):
        duration = cyc_cdi_controller.fit(sourcing_model=model, sourcing_periods=1000, seed=seed)
        
        start_time = datetime.now()
        avg_cost = cyc_cdi_controller.get_periodic_average_cost(sourcing_model=model, sourcing_periods=1000, seed=seed)
        end_time = datetime.now()
        time_elapsed = end_time - start_time

        total_avg_cost += avg_cost
        total_eval_duration += time_elapsed
        total_fit_duration += duration

        if avg_cost < min_cost:
            min_cost = avg_cost
            min_cost_seed = seed

    logger.info(f"Evaluated modified CDI policy for {realizations} realizations. Average cost over these realiations is {total_avg_cost/realizations:.4f}, with minimum cost {min_cost:.4f} for seed {min_cost_seed}")
    logger.info(f"Average time elapsed for training {total_fit_duration/realizations} and Average time elapsed for evaluation {total_eval_duration/realizations}")

if __name__ == "__main__":

    calculate_periodic_cdi_avg_cost()