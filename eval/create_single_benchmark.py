"""
Script to evaluate the Dual Source Model with various hparams
"""
import os
from tqdm import tqdm
import torch
from idinn.sourcing_model import DualSourcingModel
from idinn.dual_controller import DualSourcingNeuralController
from idinn.demand import UniformDemand
import json

json_file = "results.json"

if not os.path.exists(json_file):
    with open(json_file, 'w') as f:
        json.dump({}, f, indent=4)

with open(json_file, 'r') as f:
    data = json.load(f)

# Sourcing model is fixed - for a given lr, ce, b and U(0,a).
# The paper reports the best model cost (for the above case) to be - 23.13
dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2, # lr
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20, # ce
    holding_cost=5,
    shortage_cost=495, # b
    batch_size=256,
    init_inventory=6,
    demand_generator=UniformDemand(low=0, high=4), # U(0,4)
)


# These are the variables - We still have a different cost for each run
# But the SD is largely reduced with the fixes in Seed

activations = [torch.nn.Softplus(), torch.nn.ReLU(), torch.nn.ReLU6(), # Max val is 5, Relu 6 should work?
               "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"] # custom LogReLU variants

N = 5 # Run 10 instances per

for i in range(N):
    print("*"*100)
    print(f"RUN {i}")
    print("*"*100)
    for activation in tqdm(activations):

        print(type(activation))
        if type(activation) == torch.nn.modules.activation.Softplus:
            id_val = "softplus"
        elif type(activation) == torch.nn.modules.activation.ReLU:
            id_val = "relu"
        elif type(activation) == torch.nn.modules.activation.ReLU6:
            id_val = "relu6"
        else:
            id_val = activation

        custom_id = id_val + "_" + str(i)

        if custom_id not in data:
            controller_neural = DualSourcingNeuralController(
                hidden_layers=[128, 64, 32, 16, 8, 4],
                activation=torch.nn.CELU(alpha=1),
                expedited_activation=torch.nn.ReLU(),
                regular_activation=activation, # Change this
            )

            reg_q, exp_q = controller_neural.fit(
                sourcing_model=dual_sourcing_model,
                sourcing_periods=100, # 100 actual, in validation - 1k
                validation_sourcing_periods=1000,
                epochs=2000, #2k-3k actual
                # tensorboard_writer=SummaryWriter(comment="dual"),
                seed=123,
            )

            avg_cost = controller_neural.get_average_cost(dual_sourcing_model, sourcing_periods=1000, seed=123)
            print(f"Average cost: {avg_cost:.2f} for activation: {activation}")

            datapoint = {
                "reg_q": reg_q,
                "exp_q": exp_q,
                "avg_cost": avg_cost,
                "N": i
            }

            data[custom_id] = datapoint

            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)