import torch
from idinn.sourcing_model import DualSourcingModel
from idinn.demand import UniformDemand
from idinn.dual_controller.dual_neural_3day import DualSourcingNeuralController

def main():
    # ---------------------------
    # Problem / model parameters
    # ---------------------------
    # Lead times: set both to 0 (orders arrive same day)
    regular_lead_time = 0
    expedited_lead_time = 0

    # Costs (you provided these; keep or change as needed)
    regular_order_cost = 0.0     # cr (per-unit regular)
    expedited_order_cost = 20.0  # ce (per-unit expedited)
    holding_cost = 5.0           # h
    shortage_cost = 495.0        # b

    batch_size = 256
    # Initial inventory (learnable parameter inside the sourcing model)
    init_inventory = 6.0
    # Demand generator: discrete uniform from 0..4
    demand_generator = UniformDemand(low=0, high=4)

    # ---------------------------
    # Build the DualSourcingModel
    # ---------------------------
    dual_sourcing_model = DualSourcingModel(
        regular_lead_time=regular_lead_time,        # required arg
        expedited_lead_time=expedited_lead_time,    # required arg
        regular_order_cost=regular_order_cost,      # cost per unit regular
        expedited_order_cost=expedited_order_cost,  # cost per unit expedited
        holding_cost=holding_cost,                  # holding cost
        shortage_cost=shortage_cost,                # shortage/backlog cost
        init_inventory=init_inventory,              # learnable init inventory
        demand_generator=demand_generator,          # demand sampler
        batch_size=batch_size,                      # batch size (must match controller usage)
    )

    # ---------------------------
    # Build the controller
    # ---------------------------
    # NOTE: `use_ste` is NOT an argument in the new controller — STE is applied internally.
    controller_neural = DualSourcingNeuralController(
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.CELU(alpha=1),       # body activation
        expedited_activation=torch.nn.ReLU(),    # activation used on expedited day outputs
        regular_activation=torch.nn.ReLU(),      # activation used on regular day output
        compressed=False,                        # keep default input style
    )

    # ---------------------------
    # Fit hyperparameters
    # ---------------------------
    sourcing_periods = 100     # number of simulated 3-subday cycles per epoch
    epochs = 3000               # reduced for demo; set to 3000 for full run if you want
    validation_sourcing_periods = 200  # lower for quick tests; increase for stable validation
    validation_freq = 50
    seed = 123

    # # Choose device: CUDA → MPS → CPU
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")  # Apple Silicon (M1/M2/M3) GPU
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")

    print("Using device:", device)

    # ---------------------------
    # Run training (fit)
    # ---------------------------
    # NOTE: controller.fit internally calls controller.get_total_cost that unrolls
    # three sub-days and uses sourcing_model.order() to simulate each sub-day.
    controller_neural.fit(
        sourcing_model=dual_sourcing_model,
        sourcing_periods=sourcing_periods,
        epochs=epochs,
        validation_sourcing_periods=validation_sourcing_periods,
        validation_freq=validation_freq,
        log_freq=50,                # how often to log epoch info
        init_inventory_freq=4,      # keep learning init inventory occasionally
        init_inventory_lr=1e-1,     # lr for init inventory
        parameters_lr=3e-3,         # lr for NN parameters
        seed=seed,
        device=device,
    )


if __name__ == "__main__":
    main()
