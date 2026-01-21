
from idinn.demand import UniformDemand
from idinn.demand_three_subperiods import DiscreteTruncatedGammaDemand
from idinn.dual_controller import DynamicProgrammingController
from idinn.rail_road_model import RailRoadInventoryModel, RailRoadInventoryModelWithLeadTime
from idinn.rail_road_controller.dynamic_programming import RailRoadDPController
from idinn.sourcing_model import DualSourcingModel
from idinn.rail_road_controller.dynamic_programming_leadtime import RailRoadDPControllerWithLeadTime

def get_dual_sourcing_cost():
    dual_sourcing_model_train = DualSourcingModel(
            regular_lead_time=2,
            expedited_lead_time=0,
            regular_order_cost=0,
            expedited_order_cost=20,
            holding_cost=5,
            shortage_cost=495,
            init_inventory=0,
            demand_generator=DiscreteTruncatedGammaDemand(mean=6, std=3.4, d_max=4), # ~13.945
            # demand_generator=UniformDemand(0,4), # ~23.07
        )
    controller_dp = DynamicProgrammingController()
    controller_dp.fit(dual_sourcing_model_train, max_iterations=200_000, tolerance=1e-6, validation_freq=100,)

    avg_cost = controller_dp.get_average_cost(
        dual_sourcing_model_train, sourcing_periods=1000, seed=42
    )

    print(f"Average cost for Dual Sourcing Model: {avg_cost}")


def get_rail_road_cost():
    rail_road_model_train = RailRoadInventoryModel(
        cycle_length=3,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        # demand_generator=UniformDemand(0, 8),
        demand_generator=DiscreteTruncatedGammaDemand(mean=3, std=1, d_max=7),
    )

    controller_dp = RailRoadDPController()
    controller_dp.fit(rail_road_model_train, max_iterations=200_000, tolerance=1e-6, validation_freq=100,)

    avg_cost = controller_dp.get_average_cost(rail_road_model_train, periods=1000, seed=42)

    print(f"Average cost for Rail Road Model: {avg_cost}")


def get_rail_road_leadtime_cost():
    rail_road_model_train = RailRoadInventoryModelWithLeadTime(
        cycle_length=2,
        lead_time=2,                      # <-- lead time > 0
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        # demand_generator=UniformDemand(0, 8),
        demand_generator=DiscreteTruncatedGammaDemand(mean=3, std=1, d_max=7),
    )

    controller_dp = RailRoadDPControllerWithLeadTime()
    controller_dp.fit(
        rail_road_model_train,
        max_iterations=200_000,
        tolerance=1e-6,
        validation_freq=100,
    )

    avg_cost = controller_dp.get_average_cost(
        rail_road_model_train,
        periods=1000,
        seed=42,
    )

    print(f"Average cost for Rail Road Model (lead time): {avg_cost}")


if __name__ == "__main__":
    # get_dual_sourcing_cost()
    # get_rail_road_cost()
    get_rail_road_leadtime_cost()

"""
Rail Road Model = 55.38 uniform 0,4
Rail Road Model = 104.71 uniform 0,8
Rail Road Model = 68.87 Gamma - 6, 3.4, 4
Rail Road Model = 124.445 Gamma - 6, 3.4, 8

Rail Road cost = 83.78 Dp - 119.88 Neural Model

With lead times
Gamma - 3,1,7 -> 
"""