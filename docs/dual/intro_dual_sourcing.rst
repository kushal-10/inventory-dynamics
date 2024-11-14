Introduction to Dual-Sourcing Problems
======================================

Dual-sourcing problems are similar to single-sourcing problems but are more intricate. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering varying lead times (the duration for orders to arrive) and order costs (the expense of placing an order). The challenge lies in the company's decision-making process: determining which supplier to engage for each product to minimize costs given stochastic demand. We can solve dual-sourcing problems with `idinn` in a way similar to the approaches described in :doc:`/get_started/get_started` and :doc:`/tutorials/dual`.

We introduce the following notation to formulate the problem.

:math:`I_t`: Net inventory before replenishment in period :math:`t`.

:math:`D_t`: Demand in period :math:`t`.

:math:`b, h`: Backlogging and holding costs.

:math:`q^{\rm r}_t, q^{\rm e}_t`: Quantities ordered from the regular and expedited suppliers in period :math:`t`, respectively.

:math:`c_{\rm r}, c_{\rm e}`: Ordering costs from the regular and expedited suppliers, respectively.

:math:`l_{\rm r}, l_{\rm e}`: Lead times of the regular and expedited suppliers, respectively.

The net inventory evolves according to

.. math::

   I_{t+1} = I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t \,,

and the cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c^{\rm r} q^{\rm r}_t + c^{\rm e} q^{\rm e}_t + h \max(0, I_t) + b \max(0, - I_t)\,.

