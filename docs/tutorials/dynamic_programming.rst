Dynamic Programming
===================

We can solve dual-sourcing problems with `idinn` in similar API as other controllers using `DynamicProgrammingController`.

The Bellman Equation
-------
Dual-sourcing problems can be formulated and solved via Dynamic Programming, using the Bellman Equation. However, because of the curse of dimensionality, the ability of this approach to solve large-scale problems is limited. In what follows, we introduce some necessary notation to formulate the problem.


Example Use
-------


In this tutorial, we examine a dual-sourcing model characterized by the following parameters: the regular order lead time is 2; the expedited order lead time is 0; the regular order cost, :math:`c^r`, is 0; the expedited order cost, :math:`c^e`, is 20; and the initial inventory is 6. Additionally, the holding cost, :math:`h`, and the shortage cost, :math:`b`, are 5 and 495, respectively. Demand is generated from a discrete uniform distribution with support :math:`[1, 4]`. In this example, we use a batch size of 256. 

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import DualSourcingModel
   from idinn.controller import DualSourcingNeuralController
   from idinn.demand import UniformDemand

   dual_sourcing_model = DualSourcingModel(
       regular_lead_time=2,
       expedited_lead_time=0,
       regular_order_cost=0,
       expedited_order_cost=20,
       holding_cost=5,
       shortage_cost=495,
       batch_size=256,
       init_inventory=6,
       demand_generator=UniformDemand(low=1, high=4),
   )

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c^r q^r_t + c^e q^e_t + h \max(0, I_t) + b \max(0, - I_t)\,,

where :math:`I_t` is the inventory level at the end of period :math:`t`, :math:`q^r_t` is the regular order placed at period :math:`t`, and :math:`q^e_t` is the expedited order placed at period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory positive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The higher the regular and expedited order costs, the more costly it is to place the respective orders. The joint holding and stockout cost across all periods can be can be calculated using the `get_total_cost()` method of the sourcing model.