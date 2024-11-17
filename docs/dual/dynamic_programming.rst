Dynamic Programming
===================

Dual-sourcing problems can be formulated and solved via dynamic programming, using the Bellman equation. However, because of the curse of dimensionality, the ability of this approach to solve large-scale problems is limited.

Mathematical Structure
----------------------

The current implementation assumes :math:`l_e=c_r=0`. Note that :math:`c_r=0` can be assumed without loss of generality, while :math:`l_e=0` implies `some` loss of generality, but allows our implementation to be more versatile because otherwise we would need an overhead of calculating demand convolutions. As such, we hereafter set :math:`l_r=l` for notational convenience.


Finally, in order to introduce the Bellman Equation, we further define the following quantities:

:math:`I_t^e=I_t+q_{t-l}`: Expedited inventory position.


**States:** :math:`\mathbf{s}_t=(I_t^e, q^r_{t-l+1}, \dots, q^r_{t-1})`. 


The state space is denoted as the set of feasible states, :math:`\mathcal{S}=\{\mathbf{s}\}`.


**Actions:** :math:`\mathbf{Q}=(q^r,q^e)`. We define the action space as :math:`\mathcal{D}_\mathbf{Q}:=\{\mathbf{Q}\}`.


*Note*: In theory, both the state and the action spaces are infinitely countable. For example, the policy of not placing any orders will cause the state to approach :math:`(-\infty, 0, \dots, 0)`. Likewise, since there is no limit on the order sizes, the action space is a subset of :math:`\mathbb{N}^2`. However, we restrict our attention to static, near-optimal policies, which in the steady state generate an ergodic markov chain in the state space. This means we can restrict our attention to a finite part of the state and action spaces.

**Cost:** The cost function is :math:`f(x)=b[-x]^++h[x]^+`, where :math:`[x]^+=\max\{x,0\}`


**Transition**
Once we have selected the actions :math:`(q^r_t,q^e_t)`, random demand :math:`D_t` is realized in period :math:`t`. Then the state :math:`\mathbf{s}_t=(I_t^e, q^r_{t-l+1}, \dots, q^r_{t-1})` experiences the following transitions:

.. math::

   I^e_{t+1} &\leftarrow I^e_{t}+q^e_t+q^r_{t-l+1}-D_t\\
   q^r_{t-l+1} &\leftarrow q^r_{t-l+1}\\
   &\dots\\
   q^r_{t-2} &\leftarrow q^r_{t-1}\\
   q^r_{t-1}&\leftarrow q^r_t

The Bellman Equation is as follows:

.. math::
   J_{t+1}(\mathbf{s})=\min_{\mathbf{a}_t\in \mathcal{A}_t}\left\{c_t(\mathbf{s}_t,\mathbf{a}_t)+\gamma \sum_{\mathbf{s}'\in\mathcal{S}_{t}} \Pr(\mathbf{s}_{t+1}=\mathbf{s'}|\mathbf{s}_t,\mathbf{a}_t)J_{t}(\mathbf{s}')\right\},\,  \mathbf{s}\in\mathcal{S}.


Using renewal theory, it can be shown that for stationary demand distributions 

.. math::
   J^*=\lim\limits_{t\rightarrow \infty}\frac{J_t(\mathbf{s})}{t}


The implemented Dynamic Programming controller solves the Bellman Equation using Value Iteration.
The iterations are as follows:
 - For each state :math:`\mathbf{s} \in \mathcal{S}`, select an arbitrary initial cost :math:`J_0(\mathbf{s})`.
 - For a given state :math:`\mathbf{s}` and action :math:`\mathbf{Q}`, find the transition probabilities to state :math:`\mathbf{s}'` according to the demand distribution :math:`\phi`. Let us denote those probabilities by :math:`P(\mathbf{s}' | \mathbf{s}, \mathbf{Q})`. Calculate the cost :math:`f(\mathbf{s}')` associated with each transition :math:`\mathbf{s}\xrightarrow{\mathbf{Q}} \mathbf{s}'`. Iterate those calculations for all combinations :math:`(\mathbf{s}, \mathbf{Q}) \in \mathcal{S} \times \mathcal{D}_{\mathbf{Q}}`. 
 - Apply the update:
.. math::
   J_{k+1}(\mathbf{s}) = \min\limits_{\mathbf{Q} \in \mathcal{D}_{\mathbf{Q}}} \left\{ c_{\rm e}q^{\rm e} + \sum\limits_{\mathbf{s}' \in \mathcal{S}} P(\mathbf{s}' | \mathbf{s}, \mathbf{Q})(f(\mathbf{s}')+J_{k}(\mathbf{s}')) \right\}, \text{for all } \quad \mathbf{s} \in \mathcal{S}
- Calculate the expected cost approximation :math:`\lambda_{k+1}(\mathbf{s}) = J_{k+1}(\mathbf{s}) / (k+1)`, for all :math:`\mathbf{s} \in \mathcal{S}`
- Iterate the above update until :math:`\max\limits_{\mathbf{s}\in\mathcal{S}}\left\{\lambda_{k+1}(\mathbf{s})-\lambda_{k}(\mathbf{s})\right\} < \epsilon`


Example Usage
-------------

We can solve dual-sourcing problems with `idinn` in similar API as other controllers using `DynamicProgrammingController`.

In this example, we examine a dual-sourcing model characterized by the following parameters: 
   - Regular order lead time  :math:`l=2` 
   - Expedited order lead time :math:`l^e=0` 
   - Regular order cost :math:`c^r=0` 
   - Expedited order cost :math:`c^e=20`
   - Holding cost :math:`h=5`, shortage cost :math:`b=495`
   - Demand is generated from a discrete uniform distribution with support :math:`[1, 4]`


.. code-block:: python

   from idinn.sourcing_model import DualSourcingModel
   from idinn.dual_controller import dynamic_programming
   from dynamic_programming import DynamicProgrammingController

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

   dp_controller = DynamicProgrammingController()

   dp_controller.fit(dual_sourcing_model)


Draft below - continue here!

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