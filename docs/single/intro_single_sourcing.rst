Introduction to Single-Sourcing Problems
========================================

The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize inventory-related costs, given stochastic demand. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost :math:`h`. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost :math:`b`. 

The optimal solution to this problem is the so-called :doc:`base-stock policy </single/base_stock>`. This problem can also be addressed using the :doc:`neural network controllers </single/single_neural>` available in `idinn`.

We use the following notation to formulate the problem.

:math:`I_t`: Net inventory before replenishment in period :math:`t`.

:math:`D_t`: Demand in period :math:`t`.

:math:`b, h`: Backlogging and holding costs.

:math:`q_t`: Quantity ordered from the supplier in period :math:`t`.

:math:`c`: Ordering cost from the supplier.

:math:`l`: Supplier lead time.