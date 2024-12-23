import pytest
import torch

from idinn.demand import UniformDemand
from idinn.single_controller import BaseStockController
from idinn.sourcing_model import SingleSourcingModel


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
def base_stock_controller():
    return BaseStockController()


def test_base_stock_controller(
    single_sourcing_model: SingleSourcingModel,
    base_stock_controller: BaseStockController,
):
    # Fit the controller to the model
    base_stock_controller.fit(single_sourcing_model, seed=42)
    assert base_stock_controller.z_star == 11, "z_star value is incorrect"

    # Validate z_star calculation
    lead_time = single_sourcing_model.get_lead_time()
    num_samples = 100000
    samples = single_sourcing_model.demand_generator.sample(
        batch_size=num_samples, batch_width=lead_time + 1
    )
    total_demand_samples = samples.sum(dim=1)
    b = single_sourcing_model.get_shortage_cost()
    h = single_sourcing_model.get_holding_cost()
    expected_service_level = b / (b + h)
    expected_z_star = torch.quantile(
        total_demand_samples.float(), expected_service_level
    )
    assert (
        abs(base_stock_controller.z_star - expected_z_star) < 0.1
    ), f"z_star value deviates from expected: {expected_z_star}"

    # Calculate average cost
    avg_cost = base_stock_controller.get_average_cost(
        single_sourcing_model, sourcing_periods=1000, seed=42
    )
    assert (
        abs(avg_cost - 29) < 1
    ), f"Average cost is not within acceptable range: {avg_cost}"

    # Simulate
    past_inventories, past_orders = base_stock_controller.simulate(
        single_sourcing_model, sourcing_periods=100
    )
    assert (
        len(past_inventories) == 101
    ), "Simulation did not return correct inventory records"
    assert len(past_orders) == 101, "Simulation did not return correct order records"

    # Plot
    base_stock_controller.plot(single_sourcing_model, sourcing_periods=100)

    # Predict
    predict = base_stock_controller.predict(current_inventory=10)
    assert predict == torch.tensor(
        [[1.0]]
    ), f"Expected predict to be 1.0, but got {predict}"


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


def test_base_stock_controller_with_zero_lead_time(
    single_sourcing_model: SingleSourcingModel,
    base_stock_controller: BaseStockController,
):
    # Initialize the single sourcing model with zero lead time
    single_sourcing_model = SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )

    # Fit the controller to the model
    base_stock_controller.fit(single_sourcing_model, seed=42)
    assert (
        base_stock_controller.z_star == 4
    ), f"Expected z_star to be 4, but got {base_stock_controller.z_star}"

    # Calculate the average cost
    avg_cost = base_stock_controller.get_average_cost(
        single_sourcing_model, sourcing_periods=1000, seed=42
    )
    assert abs(avg_cost - 10) < 1, f"Average cost should be near 10, but got {avg_cost}"

    # Simulate 100 sourcing periods
    past_inventories, past_orders = base_stock_controller.simulate(
        single_sourcing_model, sourcing_periods=100
    )
    assert (
        len(past_inventories) == 101
    ), "Simulation did not return correct inventory records"
    assert len(past_orders) == 101, "Simulation did not return correct order records"

    # Plot
    base_stock_controller.plot(single_sourcing_model, sourcing_periods=100)

    # Predict
    predict = base_stock_controller.predict(current_inventory=10)
    assert predict == torch.tensor(
        [[0.0]]
    ), f"Expected predict to be 0.0, but got {predict}"
