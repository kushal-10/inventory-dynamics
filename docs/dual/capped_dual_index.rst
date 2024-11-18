Capped Dual Index
=================

The capped dual index (CDI) policy is a method for managing dual sourcing in inventory, where both regular and expedited orders are used to meet demand while minimizing costs. In each period, regular orders are capped to a maximum limit and cover demand within a longer lead time, while expedited orders address immediate needs with shorter lead times. The policy relies on two target inventory levels, one for each order type, and places orders based on the difference between the target and current inventory positions. The CDI policy efficiently balances cost and responsiveness by leveraging both sourcing options.

Mathematical Structure
----------------------

The capped dual index policy (Sun & Van Mieghem, 2019) uses the following regular and expedited orders in period :math:`t`:

.. math::

   q_t^{\rm r} = \min \left\{ [{S_t^{\rm r *}} - I_t^{t+l-1}]^+, {\bar{q}_t^{\rm r *}} \right\}

and

.. math::

   q_t^{\rm e} = [{S_t^{\rm e *}} - I_t^t]^+ \,.

Here, we assume without loss of generality that :math:`l_{\rm e} = 0`. The quantity :math:`I_t^{t+k}` in the above equations denotes the sum of the net inventory level at the beginning of period :math:`t` and all in-transit orders that will arrive by period :math:`t+k`. That is,

.. math::

   I_t^{t+k} = I_{t-1} + \sum_{i=t}^{\min(t+k, t-1)} q_i^{\rm e} + \sum_{i=t-l_{\rm r}}^{t-l_{\rm r}+k} q_i^{\rm r} \,,

where :math:`k \in \{0, \dots, l_{\rm r} - 1\}`. In accordance with Sun & Van Mieghem (2019), we use the convention that :math:`\sum_{i=a}^b = 0` if :math:`a > b`. The parameters :math:`(S_t^{\rm r *}, S_t^{\rm e *}, \bar{q}_t^{\rm r *})` are found via a search procedure. If the demand distribution is time-independent, the CDI parameters are :math:`S_t^{\rm r *} \equiv S^{\rm r *}`, :math:`S_t^{\rm e *} \equiv S^{\rm e *}`, and :math:`\bar{q}_t^{\rm r *} \equiv \bar{q}^{\rm r *}`.


Example Usage
-------------

References
----------

- Sun, J., & Van Mieghem, J. A. (2019). Robust dual sourcing inventory management: Optimality of capped dual index policies and smoothing. *Manufacturing & Service Operations Management*, 21(4), 912–931.