Sourcing Models and Custom Demand
=================================

In `idinn`, we use :class:`SingleSourcingModel` and :class:`DualSourcingModel` to simulate demands, manage orders and inventory, and train controllers.

The code below shows how one can implement a single-sourcing model with a (discrete) uniform demand distribution, ranging from 0 to 4. The methods `get_past_inventories()` and `get_past_orders()` can be used to inspect the inventory and order history, respectively.

.. code-block:: python
    
    from idinn.sourcing_model import SingleSourcingModel

    single_sourcing_model = SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=0, high=4),
    )
    # Inspect inventory history
    single_sourcing_model.get_past_inventories()
    # Inspect order history
    single_sourcing_model.get_past_orders()

One of the most important parameters is the `demand_generator`, which controls the number of orders to be consumed each period. Most examples in the documentation use uniform demand, i.e. demands are uniformally distributed between the interval specified by the user. It is also possible to feed custom demand by feeding `CustomDemand` class to `demand_generator`. `CustomDemand` takes a dictionary that describes the number of orders and their repective probabilities, such as the following example:

.. code-block:: python
    
    from idinn.sourcing_model import DualSourcingModel
    from idinn.demand import CustomDemand

    sourcing_model = DualSourcingModel(
        regular_lead_time=3,
        expedited_lead_time=0,
        regular_order_cost=0,
        expedited_order_cost=20,
        holding_cost=5,
        shortage_cost=495,
        init_inventory=0,
        demand_generator=CustomDemand({5: 0.02, 6: 0.9, 7: 0.02, 8: 0.02, 9: 0.02, 10: 0.02})
    )

In this sourcing model, there is a 90% probability that the demand will be 6, and a 2% probability that the demand will be either 5, 7, 8, 9, or 10, respectively. The `CustomDemand` generator allows users to input demands customized to their specific requirements.