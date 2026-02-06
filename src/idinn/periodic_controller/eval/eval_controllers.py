import logging
import json 
import os

from src.idinn.periodic_controller.eval.base import Eval

logging.basicConfig(
    filename="src/idinn/periodic_controller/eval/evaluation.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def evaluate():
    experiment_path = os.path.join("src", "idinn", "periodic_controller", "eval", "experiments", "experiments_A.json")
    with open(experiment_path, "r") as f:
        json_data = json.load(f)

    experiments = [json_data["models"][m] for m in json_data["models"].keys()]

    for exp in experiments:
        evaluation = Eval(exp)
        avg_costs = evaluation.evaluate_controllers()
        logger.info("**"*50)
        logger.info(f"Evaluated Dual Sourcing Model with parameters {exp}")
        logger.info(f"Average costs for controllers - {avg_costs}")
        logger.info("**"*50)



if __name__ == "__main__":
    evaluate()


     

    
