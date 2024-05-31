Get Started
===========

Initialization
--------------

The basic usage of `idinn` starts with a sourcing model and a controller. First, initialize a sourcing model, such as :class:`SingleSourcingModel`, with your preferred parameters.

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel

   # Initialize the sourcing model
   sourcing_model = SingleSourcingModel(
      lead_time=0,
      holding_cost=5,
      shortage_cost=495,
      batch_size=32,
      init_inventory=10,
      demand_distribution="uniform",
      demand_low=1,
      demand_high=4
   )

Afterwards, initialize a controller that is compatible with the chosen sourcing model. In the above single-sourcing example, the relevant controller is :class:`SingleSourcingNeuralController`.

.. code-block:: python

    from idinn.controller import SingleSourcingNeuralController
    # Initialize the neural controller
    controller = SingleSourcingNeuralController()

Training
--------

The selected controller needs to be trained to find the optimal neural-network parameters.

.. code-block:: python

   # Train the neural controller
   controller.fit(
      sourcing_model=sourcing_model,
      sourcing_periods=50,
      epochs=5000
   )

Simulation, Plotting and Order Calculation
------------------------------------------

After completed training, we can inspect how the controller performs in the specified sourcing environment by plotting the inventory and order history of certain periods.

.. code-block:: python

   # Simulate and plot the results
   controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)


The trained controller can be used for optimal order quantity calculations.

.. code-block:: python

   # Calculate the optimal order quantity for applications
   controller.forward(current_inventory=10, past_orders=[1, 5])