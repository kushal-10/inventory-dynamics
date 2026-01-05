from idinn.rail_road_model import RailRoadInventoryModel
from idinn.demand import UniformDemand
from idinn.rail_road_controller.rail_road_neural import RailRoadNeuralController

def evaluate_average_cost(
    model: RailRoadInventoryModel,
    controller: RailRoadNeuralController,
    periods: int = 100_000,
    seed: int = 42,
) -> float:
    torch.manual_seed(seed)
    model.reset(batch_size=1)

    total_cost = 0.0

    for _ in range(periods):
        q = controller.predict()
        model.order(torch.tensor([[q]]))

        I = model.get_current_inventory().item()
        holding = model.get_holding_cost() * max(I, 0)
        shortage = model.get_shortage_cost() * max(-I, 0)

        c = (
            model.get_regular_order_cost()
            if model.is_regular_day()
            else model.get_expedited_order_cost()
        )

        total_cost += c * q + holding + shortage

    return total_cost / periods


demand = UniformDemand(low=0, high=4)

model = RailRoadInventoryModel(
    cycle_length=3,               # rail every 3 days
    regular_order_cost=0.0,       # cheap rail
    expedited_order_cost=20.0,    # expensive road
    holding_cost=5.0,
    shortage_cost=495.0,
    init_inventory=6,
    demand_generator=demand,
    batch_size=1,
)



controller = RailRoadNeuralController(
    cycle_length=3,
    trunk_layers=[64, 64],
    head_layers=[32, 16],
)

controller.fit(
    model=model,
    periods=200,          # rollout length per epoch
    epochs=2000,          # training epochs
    parameters_lr=3e-3,
    init_inventory_lr=1e-1,
    seed=42,
)

avg_cost = evaluate_average_cost(
    model,
    controller,
    periods=100_000,
    seed=123,
)

print(f"Average cost (NN): {avg_cost:.4f}")
