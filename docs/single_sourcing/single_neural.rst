Single-Sourcing Neural Network Controller
=========================================

Rather than adopting a dynamic programming approach, we can parameterize actions using neural networks. 

For further details, see Böttcher, Asikis, and Fragkos (2023).

Example Usage
-------------

.. code-block:: python
    
    from idinn.sourcing_model import SingleSourcingModel
    from idinn.single_controller import SingleSourcingNeuralController
    from idinn.demand import UniformDemand
    from torch.utils.tensorboard import SummaryWriter

    single_sourcing_model = SingleSourcingModel(
        lead_time=0,
        holding_cost=5,
        shortage_cost=495,
        batch_size=32,
        init_inventory=10,
        demand_generator=UniformDemand(low=1, high=4),
    )
    controller_neural = SingleSourcingNeuralController()
    controller_neural.fit(
        sourcing_model=single_sourcing_model,
        sourcing_periods=50,
        validation_sourcing_periods=1000,
        epochs=2000,
        tensorboard_writer=SummaryWriter(comment="_single_1"),
        seed=1,
    )
    # Avg. cost 7.5725
    controller_neural.get_average_cost(single_sourcing_model, sourcing_periods=1000)

For prediction, note that the `past_orders` is optional, depending on the lead time. The controller ... 

.. code-block:: python

    # Calculate the optimal order quantity for applications
    controller.predict(current_inventory=10, past_orders=[1, 5])