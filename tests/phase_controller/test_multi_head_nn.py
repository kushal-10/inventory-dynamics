import logging
import torch
from tqdm import tqdm 

<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
from src.idinn.phase_controller.neural.multi_period_controller import MultiPeriodNeuralController
========
from src.idinn.phase_controller.neural.multi_head_controller import MultiPeriodNeuralControllerV2
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
from src.idinn.sourcing_model import DualSourcingModel
from src.idinn.demand import UniformDemand

# ---------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------
logging.basicConfig(
<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
    filename="tests/phase_controller/test_multi_period.log",
========
    filename="tests/phase_controller/test_multi_head.log",
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


sourcing_model = DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=6,
        demand_generator=UniformDemand(0, 4),
        batch_size=1,
    )

<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
def test_multi_period_model():
    
    controller = MultiPeriodNeuralController(
        hidden_layers=[128, 64, 32, 16, 8],
========
def test_multi_head_model():
    
    controller = MultiPeriodNeuralControllerV2(
        hidden_layers=[256, 128, 64, 32, 16, 8],
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
        n_periods=2
    )

    controller.fit(
        sourcing_model=sourcing_model,
<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
        sourcing_periods=100,
        epochs=500,
========
        sourcing_periods=300,
        epochs=500,
        seed=42,
        parameters_lr=5e-4,
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
    )

    # multi-seed evaluation
    costs = []
    with torch.no_grad():
<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
        for seed in tqdm(range(50)):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=1000,
========
        for seed in tqdm(range(500)):
            cost = controller.get_average_cost(
                sourcing_model=sourcing_model,
                sourcing_periods=100,
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
                seed=seed,
            )
            costs.append(cost)

    mean_cost = torch.mean(torch.stack(costs))
    std_cost = torch.std(torch.stack(costs))

    logger.info(f"Final mean cost: {mean_cost}")
    logger.info(f"Final std cost: {std_cost}")

    print("Final mean:", mean_cost.item())
    print("Final std:", std_cost.item())



if __name__ == '__main__':
<<<<<<<< HEAD:tests/phase_controller/test_multi_period_nn.py
    test_multi_period_model()
========
    test_multi_head_model()
>>>>>>>> main:tests/phase_controller/test_multi_head_nn.py
