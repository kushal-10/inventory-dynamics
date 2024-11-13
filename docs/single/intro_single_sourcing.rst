Single-Sourcing Problems
========================

The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize inventory-related costs, given stochastic demand. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost :math:`h`. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost :math:`b`. This problem can be addressed using `idinn`. As illustrated in the :doc:`/get_started/get_started` section, we first initialize the sourcing model and its associated neural network controller. Subsequently, we train the neural network controller using data generated from the sourcing model. Finally, we use the trained neural network controller to compute optimal order quantities, which depend on the state of the system.

In what follows, we introduce some necessary notation to formulate the problem.

:math:`I_t`: net inventory before replenishment in period :math:`t`.

:math:`D_t`: demand in period :math:`t`.

:math:`b, h`: backlogging and holding costs.

:math:`q_t`: quantity ordered from the supplier in period ty.

:math:`c`: ordering cost from the supplier.

:math:`l`: lead time of the supplier.