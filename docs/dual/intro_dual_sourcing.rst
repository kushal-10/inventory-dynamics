Introduction to Dual-Sourcing Problems
======================================

Dual-sourcing problems are similar to single-sourcing problems but are more intricate. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering varying lead times (the duration for orders to arrive) and order costs (the expense of placing an order). The challenge lies in the company's decision-making process: determining which supplier to engage for each product to minimize costs given stochastic demand.

The optimal solution to this problem is unknown. However, various heuristics have been developed, and the :doc:`capped dual-index policy </dual/capped_dual_index>` has demonstrated strong performance across a wide range of parameters. Another approach is to approximate the optimal solution using :doc:`dynamic programming </dual/dynamic_programming>`. Lastly, this problem can also be addressed using the :doc:`neural network controllers </dual/dual_neural>` available in `idinn`.

We use the following notation to formulate the problem.

:math:`I_t`: Net inventory before replenishment in period :math:`t`.

:math:`D_t`: Demand in period :math:`t`.

:math:`b, h`: Backlogging and holding costs.

:math:`q^{\rm r}_t, q^{\rm e}_t`: Quantities ordered from the regular and expedited suppliers in period :math:`t`, respectively.

:math:`c_{\rm r}, c_{\rm e}`: Ordering costs from the regular and expedited suppliers, respectively.

:math:`l_{\rm r}, l_{\rm e}`: Lead times of the regular and expedited suppliers, respectively.

The sequence of events in a single period :math:`t` is as follows:

- Order quanity :math:`q_{t-l}`, ordered in period :math:`t-l`, arrives

- Order quantity :math:`q_t` is placed

- Demand :math:`D_t` is realized

- Inventory cost for the period is registered as :math:`cq_t+h(I_t+q_{t-l}-D_t)^++b(D_t-I_t-q_{t+l})^+`, where :math:`(x)^+=\max\{0, x\}`

- New state is updated as :math:`(I_t+q_{t-l}-D_t, q_{t-l+1}, q_{t-l+2},\dots,q_{t})`

The net inventory evolves according to

.. math::

   I_{t+1} = I_{t} + q^{\rm r}_{t-l_{\rm r}} + q^{\rm e}_{t-l_{\rm e}} - D_t \,,

and the cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c_{\rm r} q^{\rm r}_t + c_{\rm e} q^{\rm e}_t + h \max(0, I_t) + b \max(0, - I_t)\,.

The higher the holding cost, the more costly it is to keep the inventory positive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The higher the regular and expedited order costs, the more costly it is to place the respective orders.