import pytest
import numpy as np
from idinn.sourcing_model import SingleSourcingModel
from idinn.single_controller import BaseStockController
from idinn.demand import UniformDemand


@pytest.fixture
def single_sourcing_model():
    return SingleSourcingModel(
        lead_time=2,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )

@pytest.fixture
def single_sourcing_model_zero_lead_time():
    return SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )

@pytest.fixture
def base_stock_controller():
    return BaseStockController()


# def test_base_stock_controller_fit(single_sourcing_model, base_stock_controller):
#     base_stock_controller.fit(single_sourcing_model)
    # assert hasattr(controller, "sourcing_model"), "sourcing_model should be set after fitting"
    # assert hasattr(controller, "z_star"), "z_star should be set after fitting"
    # assert controller.z_star == pytest.approx(11, abs=0.5), "z_star should be close to 11"


# def test_base_stock_controller_predict():
#     controller = BaseStockController()
#     controller.z_star = 11

#     # Test when inventory position is below z_star
#     action = controller.predict(5, np.array([1, 2, 3]))
#     assert action == 6, "Should order up to z_star"

#     # Test when inventory position is above z_star
#     action = controller.get_action(15, np.array([1, 2, 3]))
#     assert action == 0, "Should not order when inventory position is above z_star"


# def test_base_stock_controller_reset():
#     controller = BaseStockController()
#     controller.fit(single_sourcing_model)

#     controller.reset()
#     assert not hasattr(controller, "z_star"), "z_star should be removed after reset"
#     assert not hasattr(
#         controller, "sourcing_model"
#     ), "sourcing_model should be removed after reset"


# def test_base_stock_controller_cost(single_sourcing_model):
#     controller = BaseStockController()
#     controller.fit(single_sourcing_model)

#     total_cost = controller.get_total_cost(
#         single_sourcing_model, sourcing_periods=1000
#     ).mean()
#     assert total_cost == pytest.approx(
#         29000, rel=0.1
#     ), "Total cost should be near 29000"

#     average_cost = controller.get_average_cost(
#         single_sourcing_model, sourcing_periods=1000
#     )
#     assert average_cost == pytest.approx(29, rel=0.1), "Average cost should be near 29"


def test_base_stock_controller_zero_lead_time_fit(single_sourcing_model_zero_lead_time):
    controller = BaseStockController()
    controller.fit(single_sourcing_model_zero_lead_time)

    assert controller.z_star == pytest.approx(
        4, abs=0.5
    ), "z_star should be close to 4 for zero lead time"


# def test_base_stock_controller_zero_lead_time_cost(single_sourcing_model):
#     controller = BaseStockController()
#     controller.fit(single_sourcing_model)

#     total_cost = controller.get_total_cost(
#         single_sourcing_model, sourcing_periods=1000
#     ).mean()
#     assert total_cost == pytest.approx(
#         10000, rel=0.1
#     ), "Total cost should be near 10000"

#     average_cost = controller.get_average_cost(
#         single_sourcing_model, sourcing_periods=1000
#     )
#     assert average_cost == pytest.approx(10, rel=0.1), "Average cost should be near 10"
