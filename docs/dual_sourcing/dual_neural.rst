Dual-Sourcing Neural Network Controller
=======================================

Rather than adopting a dynamic programming approach, we can parameterize actions using a neural network. The optimization process is illustrated schematically in the figure below. The states :math:`\{\mathbf{s}_t^{(j)}\}` (:math:`j \in \{1, \dots, M\}`), which evolve according to the underlying discrete-time dynamics, are used as inputs to a neural network. This network is trained to produce actions that minimize the expected cost per period.

.. image:: ../_static/optimization_schematic.png
   :alt: Neural-network optimization process
   :align: center

For further details, see Böttcher, Asikis, and Fragkos (2023).

Example Usage
--------------

We now present one example to demonstrate how the :class:`DualSourcingNeuralController` can be called, trained, and evaluated in `idinn`.

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import DualSourcingModel
   from idinn.dual_controller import DualSourcingNeuralController
   from idinn.demand import UniformDemand
   from torch.utils.tensorboard import SummaryWriter

   dual_sourcing_model = DualSourcingModel(
       regular_lead_time=2,
       expedited_lead_time=0,
       regular_order_cost=0,
       expedited_order_cost=20,
       holding_cost=5,
       shortage_cost=495,
       batch_size=256,
       init_inventory=6,
       demand_generator=UniformDemand(low=0, high=4),
   )
   controller_neural = DualSourcingNeuralController(
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.CELU(alpha=1)
    )
    controller_neural.fit(
        sourcing_model=dual_sourcing_model,
        sourcing_periods=100,
        validation_sourcing_periods=1000,
        epochs=2000,
        tensorboard_writer=SummaryWriter(comment="dual"),
        seed=123,
    )
    # Avg. cost near 24
    controller_neural.get_average_cost(dual_sourcing_model, sourcing_periods=1000)

Adjusting parameters such as `batch_size`, `init_inventory`, and `epochs` can improve the learning of sourcing policies. It may also be helpful to try out different neural-network structures.

For a given controller, orders can be predicted as follows.

.. code-block:: python

    controller_neural.predict(current_inventory=10, past_regular_orders=[1, 1], past_expedited_orders=None)

If the regular and expedited lead-time values are greater than 0, one has to specify the corresponding `past_regular_orders` and `past_expedited_orders`.

References
----------
- Böttcher, L., Asikis, T., & Fragkos, I. (2023). Control of dual-sourcing inventory systems using recurrent neural networks. *INFORMS Journal on Computing*, 35(6), 1308–1328.