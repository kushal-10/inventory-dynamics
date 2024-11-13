Dual-Sourcing Problems
======================

Dual-sourcing problems are similar to single-sourcing problems but are more intricate. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering varying lead times (the duration for orders to arrive) and order costs (the expense of placing an order). The challenge lies in the company's decision-making process: determining which supplier to engage for each product to minimize costs given stochastic demand. We can solve dual-sourcing problems with `idinn` in a way similar to the approaches described in :doc:`/get_started/get_started` and :doc:`/tutorials/dual`.

In what follows, we introduce some necessary notation to formulate the problem.

:math:`I_t`: net inventory before replenishment in period :math:`t`.

:math:`D_t`: demand in period :math:`t`.

:math:`b, h`: backlogging and holding costs.

:math:`q^r_t, q^e_t`: quantity ordered from the regular or expedited supplier in period t, respectively.

:math:`c_r, c_e`: ordering cost from the regular or expedited supplier, respectively.

:math:`l_r, l_e`: lead time of the regular or expedited supplier, respectively.

define cost and inventory evolution.

.. math::

   I_{t+1} = I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t \,,

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c^r q^r_t + c^e q^e_t + h \max(0, I_t) + b \max(0, - I_t)\,,

