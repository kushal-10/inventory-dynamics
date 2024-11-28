Single-Sourcing Neural Network Controller
=========================================

Rather than using a base-stock controller, we can parameterize actions using a neural network. This network is trained to generate actions that minimize the expected cost per period for a single-sourcing system that evolves according to its discrete-time dynamics.

For further details, see Böttcher, Asikis, and Fragkos (2023).

Example Usage
-------------

We now present one example to demonstrate how the `SingleSourcingNeuralController` can be called, trained, and evaluated in `idinn`.

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
        demand_generator=UniformDemand(low=0, high=4),
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
    # Avg. cost near 10
    controller_neural.get_average_cost(single_sourcing_model, sourcing_periods=1000)

Adjusting parameters such as `batch_size` and `epochs` can improve the learning of sourcing policies.

For a given controller, orders can be predicted as follows.

.. code-block:: python

    # Calculate the optimal order quantity for applications
    controller_neural.predict(current_inventory=10, past_orders=[])

If the lead-time value is greater than 0, one has to specify the corresponding `past_orders`.

References
----------
- Böttcher, L., Asikis, T., & Fragkos, I. (2023). Control of dual-sourcing inventory systems using recurrent neural networks. *INFORMS Journal on Computing*, 35(6), 1308–1328.