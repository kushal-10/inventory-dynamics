Sourcing Models and Custom Demand
=================================

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
        demand_generator=CustomDemand({5: 0.2, 6: 0.3, 7: 0.1, 8: 0.1, 9: 0.1})
    )