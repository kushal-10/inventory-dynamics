Base Stock
==========

The base stock policy for single-sourcing problems is an inventory control approach where a fixed target inventory level, or "base stock level," is maintained. Whenever inventory drops below this level due to demand, a replenishment order is placed to bring it back up to the target. This policy balances holding costs (by limiting excess stock) and stockout costs (by ensuring enough inventory to meet demand) and is ideal for products with consistent demand. The target level is typically set based on forecasted demand, lead times, and acceptable service levels.

Mathematical Structure
----------------------

To mathematically describe the optimal order policy of single-sourcing problems :citep:`arrow1951optimal, scarf1958inventory`, we use :math:`l` and :math:`z` to respectively denote the replenishment lead time and the target inventory-position level (i.e., the target net inventory level plus all goods on order). The inventory position of single-sourcing dynamics at time :math:`t`, :math:`\tilde{I}_t`, is given by

.. math::

   \tilde{I}_t =
   \begin{cases}
      I_t & \text{if} \,\, l=0 \\
      I_t + \sum_{i=1}^l q_{t-i} & \text{if} \,\, l>0 \,,
   \end{cases}

where :math:`I_t` and :math:`q_t` denote the net inventory at time :math:`t` and the replenishment order placed at time :math:`t`, respectively. We


Example Usage
-------------

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.controller import SingleSourcingNeuralController
   from idinn.demand import UniformDemand

   single_sourcing_model = SingleSourcingModel(
       lead_time=0,
       holding_cost=5,
       shortage_cost=495,
       batch_size=32,
       init_inventory=10,
       demand_generator=UniformDemand(low=1, high=4),
    )