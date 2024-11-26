Introduction
============

The overall objective in single-sourcing and related inventory management problems is to identify the optimal order quantities to minimize expected inventory-related costs, given stochastic demand over a finite or inifinite time horizon. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost :math:`h`. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost :math:`b` per unit. 

The optimal solution to this problem is the so-called :doc:`base-stock policy </single/base_stock>`. This problem can also be addressed using the :doc:`neural network controllers </single/single_neural>` available in `idinn`.

We use the following notation to formulate the problem.

:math:`I_t`:  Net inventory before replenishment in period :math:`t`.

:math:`D_t`:  Demand in period :math:`t`.

:math:`b, h`: Unit backlogging and holding costs.

:math:`q_t`:  Quantity ordered from the supplier in period :math:`t`.

:math:`l`:  Supplier lead time.

The sequence of events in a single period :math:`t` is as follows:

- Order quanity :math:`q_{t-l}`, ordered in period :math:`t-l`, arrives

- Order quantity :math:`q_t` is placed

- Demand :math:`D_t` is realized

- Inventory cost for the period is registered as :math:`h[I_t+q_{t-l}-D_t]^++b[D_t-I_t-q_{t+l}]^+`, where :math:`[x]^+=\max\{0, x\}`

- New state is updated as :math:`(I_t+q_{t-l}-D_t, q_{t-l+1}, q_{t-l+2},\dots,q_{t})`

The net inventory evolves according to

.. math::

   I_{t+1} = I_{t} + q_{t-l} - D_t \,,

and the cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = h \max(0, I_{t+1}) + b \max(0, - I_{t+1})\,.

The higher the holding cost, the more costly it is to keep the inventory postive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. 