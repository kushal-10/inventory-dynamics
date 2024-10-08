import torch
import pytest
from idinn.sourcing_model import SingleSourcingModel, DualSourcingModel
from idinn.demand import CustomDemand


@pytest.fixture
def single_sourcing_model():
    """
    Single sourcing model at default state.
    """
    return SingleSourcingModel(
        lead_time=2,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
    )


def test_single_sourcing_model_get_lead_time(
    single_sourcing_model: SingleSourcingModel,
):
    assert single_sourcing_model.get_lead_time() == 2


def test_single_sourcing_model_get_init_inventory(
    single_sourcing_model: SingleSourcingModel,
):
    assert torch.all(
        torch.eq(single_sourcing_model.get_init_inventory(), torch.tensor([10.0]))
    )


def test_single_sourcing_model_get_past_orders(
    single_sourcing_model: SingleSourcingModel,
):
    assert torch.all(
        torch.eq(single_sourcing_model.get_past_orders(), torch.zeros(3, 3))
    )


def test_single_sourcing_model_get_past_inventories(
    single_sourcing_model: SingleSourcingModel,
):
    assert torch.all(
        torch.eq(
            single_sourcing_model.get_past_inventories(),
            torch.tensor([10.0]).repeat(3, 3),
        )
    )


def test_single_sourcing_model_get_current_inventory(
    single_sourcing_model: SingleSourcingModel,
):
    assert torch.all(
        torch.eq(
            single_sourcing_model.get_current_inventory(),
            torch.tensor([[10.0], [10.0], [10.0]]),
        )
    )


def test_single_sourcing_model_get_cost(single_sourcing_model: SingleSourcingModel):
    assert torch.all(
        torch.eq(single_sourcing_model.get_cost(), torch.tensor([[5.0], [5.0], [5.0]]))
    )


@pytest.fixture
def single_sourcing_model_custom_demand():
    """
    Single sourcing model with fixed demand at 1.
    """
    return SingleSourcingModel(
        lead_time=1,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=CustomDemand([1, 1, 1, 1, 1]),
    )


def test_single_sourcing_model_order(
    single_sourcing_model_custom_demand: SingleSourcingModel,
):
    single_sourcing_model_custom_demand.order(torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_past_orders(),
            torch.cat([torch.zeros(3, 2), torch.tensor([[1.0], [2.0], [3.0]])], dim=1),
        )
    )
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_past_inventories(),
            torch.cat(
                [
                    torch.tensor([10.0]).repeat(3, 2),
                    torch.tensor([[9.0], [9.0], [9.0]]),
                ],
                dim=1,
            ),
        )
    )
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_current_inventory(),
            torch.tensor([[9.0], [9.0], [9.0]]),
        )
    )
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_cost(),
            torch.tensor([[4.5], [4.5], [4.5]]),
        )
    )

    single_sourcing_model_custom_demand.order(torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_past_orders(),
            torch.cat(
                [torch.zeros(3, 2), torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])],
                dim=1,
            ),
        )
    )
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_current_inventory(),
            torch.tensor([[9.0], [10.0], [11.0]]),
        )
    )
    assert torch.all(
        torch.eq(
            single_sourcing_model_custom_demand.get_cost(),
            torch.tensor([[4.5], [5.0], [5.5]]),
        )
    )


@pytest.fixture
def dual_sourcing_model():
    """
    Dual sourcing model at default state.
    """
    return DualSourcingModel(
        regular_lead_time=2,
        expedited_lead_time=1,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
    )


def test_dual_sourcing_model_get_expedited_lead_time(
    dual_sourcing_model: DualSourcingModel,
):
    assert dual_sourcing_model.get_expedited_lead_time() == 1


def test_dual_sourcing_model_get_regular_lead_time(
    dual_sourcing_model: DualSourcingModel,
):
    assert dual_sourcing_model.get_regular_lead_time() == 2


def test_dual_sourcing_model_get_init_inventory(dual_sourcing_model: DualSourcingModel):
    assert torch.all(
        torch.eq(dual_sourcing_model.get_init_inventory(), torch.tensor([10.0]))
    )


def test_dual_sourcing_model_get_past_regular_orders(
    dual_sourcing_model: DualSourcingModel,
):
    assert torch.all(
        torch.eq(dual_sourcing_model.get_past_regular_orders(), torch.zeros(3, 3))
    )


def test_dual_sourcing_model_get_past_expedited_orders(
    dual_sourcing_model: DualSourcingModel,
):
    assert torch.all(
        torch.eq(dual_sourcing_model.get_past_expedited_orders(), torch.zeros(3, 3))
    )


def test_dual_sourcing_model_get_past_inventories(
    dual_sourcing_model: DualSourcingModel,
):
    assert torch.all(
        torch.eq(
            dual_sourcing_model.get_past_inventories(),
            torch.tensor([10.0]).repeat(3, 3),
        )
    )


def test_dual_sourcing_model_get_current_inventory(
    dual_sourcing_model: DualSourcingModel,
):
    assert torch.all(
        torch.eq(
            dual_sourcing_model.get_current_inventory(),
            torch.tensor([[10.0], [10.0], [10.0]]),
        )
    )


def test_dual_sourcing_model_get_cost(dual_sourcing_model: DualSourcingModel):
    assert torch.all(
        torch.eq(
            dual_sourcing_model.get_cost(0, 0), torch.tensor([[5.0], [5.0], [5.0]])
        )
    )


@pytest.fixture
def dual_sourcing_model_custom_demand():
    """
    Dual sourcing model with fixed demand at 1.
    """
    return DualSourcingModel(
        regular_lead_time=1,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=0.5,
        shortage_cost=1,
        init_inventory=10,
        batch_size=3,
        demand_generator=CustomDemand([1, 1, 1, 1, 1]),
    )


def test_dual_sourcing_model_order(
    dual_sourcing_model_custom_demand: DualSourcingModel,
):
    dual_sourcing_model_custom_demand.order(
        regular_q=torch.tensor([[1.0], [2.0], [3.0]]),
        expedited_q=torch.tensor([[4.0], [5.0], [6.0]]),
    )
    assert torch.all(
        torch.eq(
            dual_sourcing_model_custom_demand.get_past_regular_orders(),
            torch.cat([torch.zeros(3, 2), torch.tensor([[1.0], [2.0], [3.0]])], dim=1),
        )
    )
    assert torch.all(
        torch.eq(
            dual_sourcing_model_custom_demand.get_past_expedited_orders(),
            torch.cat([torch.zeros(3, 2), torch.tensor([[4.0], [5.0], [6.0]])], dim=1),
        )
    )
    assert torch.all(
        torch.eq(
            dual_sourcing_model_custom_demand.get_past_inventories(),
            torch.cat(
                [
                    torch.tensor([10.0]).repeat(3, 2),
                    torch.tensor([[13.0], [14.0], [15.0]]),
                ],
                dim=1,
            ),
        )
    )
    assert torch.all(
        torch.eq(
            dual_sourcing_model_custom_demand.get_current_inventory(),
            torch.tensor([[13.0], [14.0], [15.0]]),
        )
    )
    assert torch.all(
        torch.eq(
            dual_sourcing_model_custom_demand.get_cost(regular_q=0, expedited_q=1),
            torch.tensor([[26.5], [27.0], [27.5]]),
        )
    )
