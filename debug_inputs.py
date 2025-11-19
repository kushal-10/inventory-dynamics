# run_triple_example.py
import torch
from src.idinn.three_day_sourcing_model import ThreeDayCycleModel
from src.idinn.dual_controller.triple_neural import DualSourcingNeuralController
from idinn.demand import UniformDemand  # assuming this path exists in your repo

dual_sourcing_model = ThreeDaySourcingModel(
    regular_order_cost=0.0,            # cr
    expedited_order_cost=20.0,         # ce (cost per expedited unit)
    holding_cost=5.0,                  # h
    shortage_cost=495.0,               # b
    batch_size=256,
    init_inventory=6.0,
    demand_generator=UniformDemand(low=0, high=4),  # must implement sample(batch_size)
)

controller_neural = DualSourcingNeuralController(
    hidden_layers=[8, 4],
    activation=torch.nn.CELU(alpha=1),
    expedited_activation=torch.nn.ReLU(),
    regular_activation=torch.nn.ReLU(),
    use_ste=True,
)

# Train: cycles_per_epoch=sourcing_periods argument in fit maps to cycles per epoch here
controller_neural.fit(
    sourcing_model=dual_sourcing_model,
    sourcing_periods=100,   # cycles per epoch
    epochs=3000,
    validation_sourcing_periods=1000,
    validation_freq=50,
    seed=123,
    device=torch.device("cpu"),
)
