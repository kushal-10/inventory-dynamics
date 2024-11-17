Base Stock
==========

The base-stock policy for single-sourcing problems of infinite horizon is an inventory control approach where a fixed target inventory position, or "base-stock level", is maintained. Whenever inventory drops below this level due to demand, a replenishment order is placed to bring it back up to the target. This policy balances holding costs (by limiting excess stock) and stockout costs (by ensuring enough inventory to meet demand) and is optimal for products with consistent demand in the sense that minimizes the expected (per period) inventory cost over an infinite time horizon.

Mathematical Structure
----------------------

To mathematically describe the optimal order policy of single-sourcing problems (Arrow, Harris, & Marschak, 1951; Scarf & Karlin, 1958), we use :math:`l` and :math:`z` to respectively denote the replenishment lead time and the target inventory-position level (i.e., the target net inventory level plus all items ordered but not received yet). The inventory position of single-sourcing dynamics at time :math:`t`, :math:`\tilde{I}_t`, is given by

.. math::

   \tilde{I}_t =
   \begin{cases}
      I_t & \text{if} \,\, l=0 \\
      I_t + \sum_{i=1}^l q_{t-i} & \text{if} \,\, l>0 \,,
   \end{cases}

where :math:`I_t` and :math:`q_t` denote the net inventory at time :math:`t` and the replenishment order placed at time :math:`t`, respectively. 

We can then denote the optimal order quantity as :math:`q_t=\max\{0, z-\tilde{I}_t\}`, where the target level :math:`z` is the parameter to be determined.

Example Usage
-------------

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.single_controller.single_neural import SingleSourcingNeuralController
   from idinn.demand import UniformDemand

   single_sourcing_model = SingleSourcingModel(
       lead_time=0,
       holding_cost=5,
       shortage_cost=495,
       batch_size=32,
       init_inventory=10,
       demand_generator=UniformDemand(low=1, high=4),
    )

References
----------
- Scarf, H., & Karlin, S. (1958). Inventory models of the Arrow-Harris-Marschak type with time lag. In K. J. Arrow, S. Karlin, & H. E. Scarf (Eds.), *Studies in the Mathematical Theory of Inventory and Production* (Stanford University Press, Stanford, CA).
- Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250–272.