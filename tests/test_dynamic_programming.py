from idinn.sourcing_model import DualSourcingModel
from idinn.dual_controller.dynamic_programming import (
    DynamicProgrammingController)
from idinn.demand import UniformDemand


def main():
    dual_sourcing_model = DualSourcingModel(
       regular_lead_time=2,
       expedited_lead_time=0,
       regular_order_cost=0,
       expedited_order_cost=20,
       holding_cost=5,
       shortage_cost=495,
       batch_size=256,
       init_inventory=6,
       demand_generator=UniformDemand(low=1, high=4),
    )

    dp_controller = DynamicProgrammingController()

    dp_controller.fit(dual_sourcing_model)


if __name__ == '__main__':
    main()
