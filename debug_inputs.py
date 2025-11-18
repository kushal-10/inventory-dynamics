import torch
from idinn.sourcing_model import DualSourcingModel
from idinn.dual_controller import DualSourcingNeuralController
from idinn.demand import UniformDemand
import numpy as np
from torch.utils.tensorboard import SummaryWriter


dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2, # lr
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20, # ce
    holding_cost=5,
    shortage_cost=495, # b
    batch_size=256,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4), # 23.13 ->
)


controller_neural = DualSourcingNeuralController(
    hidden_layers=[8, 4],
    activation=torch.nn.CELU(alpha=1),
    expedited_activation=torch.nn.ReLU(),
    regular_activation=torch.nn.ReLU(), # Change this
)

controller_neural.fit(sourcing_model=dual_sourcing_model, sourcing_periods=100, epochs=3000,
                      validation_sourcing_periods=1000, seed=123)