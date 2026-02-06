import logging
from typing import Dict

from tqdm import tqdm

from src.idinn.periodic_controller.multi_headed_neural_network import MultiHeadedNeuralController
from src.idinn.periodic_controller.dynamic_programming import DynamicProgrammingController
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand


# Get root logger
logger = logging.getLogger()

class Eval():

    def __init__(self, experiment: Dict = None) -> None:
        
        self.experiment = experiment
        demand=None
        if experiment["demand"] == "uniform":
            demand = UniformDemand(low=experiment["demand_low"], high=experiment["demand_high"])
        

        if demand is None:
            raise ValueError(f"Demand generator not initialized for experiment : {experiment}, check the JSON file.",
                                f"\nValid args for demand - 'uniform', got {experiment['demand']}")

        self.dual_sourcing_model = DualSourcingModel(
            init_inventory=experiment["init_inventory"],
            regular_lead_time=experiment["regular_lead_time"],
            expedited_lead_time=experiment["expedited_lead_time"],
            regular_order_cost=experiment["regular_order_cost"],
            expedited_order_cost=experiment["expedited_order_cost"],
            holding_cost=experiment["holding_cost"],
            shortage_cost=experiment["shortage_cost"],
            batch_size=experiment["batch_size"],
            demand_generator=demand
        )

        logger.info(f"Initiated controller evaluation for experiment with configuration: {experiment} ")
        
    def fit_dp(self):

        controller = DynamicProgrammingController()
        controller.fit(
            sourcing_model=self.dual_sourcing_model,
        )
        
        return controller

    def fit_nn(self):
        
        controller = MultiHeadedNeuralController(
            shared_layers=[64, 32, 16],
            head_even_layers=[8,4],
            head_odd_layers=[4]
        )

        controller.fit(
            sourcing_model=self.dual_sourcing_model,
            sourcing_periods=100,
            epochs=3000,
            weight_decay_shared = 9.563411263437158e-05,
            weight_decay_odd = 0.0007149281582778022,
            weight_decay_even = 1.2386176014952e-06,
            parameters_lr_shared = 0.0013542080013067383,
            parameters_lr_odd = 0.0008940175735593203,
            parameters_lr_even = 0.0012104440658971336,
            seed=42,
        )
    
        return controller


    def evaluate_controllers(self):

        controller_types = self.experiment["controllers"]

        average_costs = []

        for ct in tqdm(controller_types):
            logger.info(f"Evaluating controller type - {ct}")
            if ct=="dp":
                controller = self.fit_dp()
            elif ct=="nn":
                controller = self.fit_nn()

            avg_eval_cost = controller.get_periodic_average_cost(
                sourcing_model=self.dual_sourcing_model,
                sourcing_periods=self.experiment["eval_sourcing_periods"],
                seed=42
            )
            
            average_costs.append(
                {
                    ct: str(avg_eval_cost)
                }
            )

        return average_costs


        