Dual-Sourcing Neural Network Controller
=======================================

Rather than adopting a dynamic programming approach, we can parameterize actions using a neural network. The optimization process is illustrated schematically in the figure below. The states :math:`\{\mathbf{s}_t^{(j)}\}` (:math:`j \in \{1, \dots, M\}`), which evolve according to the underlying discrete-time dynamics, are used as inputs to a neural network. This network is trained to produce actions that minimize the expected cost per period.

.. image:: ../_static/optimization_schematic.png
   :alt: Neural-network optimization process
   :align: center

For further details, see Böttcher, Asikis, and Fragkos (2023).

Example Usage
--------------

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
        seed=4,
    )
    # Avg. cost 17.9469
    controller_neural.get_average_cost(dual_sourcing_model, sourcing_periods=1000)

References
----------
- Böttcher, L., Asikis, T., & Fragkos, I. (2023). Control of dual-sourcing inventory systems using recurrent neural networks. *INFORMS Journal on Computing*, 35(6), 1308–1328.