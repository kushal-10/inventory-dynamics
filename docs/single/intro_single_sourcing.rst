Introduction to Single-Sourcing Problems
========================================

The overall objective in single-sourcing and related inventory management problems is to identify the optimal order quantities to minimize expected inventory-related costs, given stochastic demand over a finite or inifinite time horizon. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost :math:`h`. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost :math:`b` per unit. 

The optimal solution to this problem is the so-called :doc:`base-stock policy </single/base_stock>`. This problem can also be addressed using the :doc:`neural network controllers </single/single_neural>` available in `idinn`.

We use the following notation to formulate the problem.

:math:`I_t`: Net inventory before replenishment in period :math:`t`.

:math:`D_t`: Demand in period :math:`t`.

:math:`b, h`: Unit backlogging and holding costs.

:math:`q_t`: Quantity ordered from the supplier in period :math:`t`.

:math:`c`: Ordering cost from the supplier.

:math:`l`: Supplier lead time.

The sequence of events in a single period :math:`t` is as follows:
- Quanity ordered in period :math:`t-l` arrives
- Order quantity :math:`q_t` is placed
- Demand :math:`D_t` is realized
- Inventory cost for the period is registered as :math:`c_tq_t+h(I_t+q_{t-l}-D_t)^+b(D_t-I_t-q_{t+l})^+`
- New state is updated as :math:`(I_t+q_{t-l}-D_t, q_{t-l+1}, q_{t-l+2},\dots,q_{t})`