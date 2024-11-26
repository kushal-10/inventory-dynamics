######################################################
idinn: Inventory-Dynamics Control with Neural Networks
######################################################

..  youtube:: hUBfTWV6tWQ
   :width: 100%

`idinn` implements inventory dynamics–informed neural networks for solving single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into customizable objects with PyTorch backend to enable users to find the optimal neural controllers for the user-specified inventory systems.

Demo
====

For a quick demo, you can run the `Streamlit app <https://idinn-demo.streamlit.app/>`_ using ``idinn``. The app allows you to interactively train and evaluate neural controllers for user-specified dual-sourcing systems.

Example Usage
=============

.. code-block:: python

   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.controller import SingleSourcingNeuralController
   from idinn.demand import UniformDemand

   # Initialize the sourcing model and the neural controller
   sourcing_model = SingleSourcingModel(
      lead_time=0,
      holding_cost=5,
      shortage_cost=495,
      batch_size=32,
      init_inventory=10,
      demand_generator=UniformDemand(low=1, high=4),
   )
   controller = SingleSourcingNeuralController(
      hidden_layers=[2],
      activation=torch.nn.CELU(alpha=1)
   )
   # Train the neural controller
   controller.fit(
      sourcing_model=sourcing_model,
      sourcing_periods=50,
      validation_sourcing_periods=1000,
      epochs=5000,
      seed=1,
   )
   # Simulate and plot the results
   controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
   # Calculate the optimal order quantity for applications
   controller.predict(current_inventory=10, past_orders=[1, 5])

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/get_started
   get_started/deployment

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Single-Sourcing Problems

   single_sourcing/intro_single_sourcing
   single_sourcing/base_stock
   single_sourcing/single_neural

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Dual-Sourcing Problems

   dual_sourcing/intro_dual_sourcing
   dual_sourcing/capped_dual_index
   dual_sourcing/dynamic_programming
   dual_sourcing/dual_neural

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Utilities

   utilities/sourcing_model
   utilities/custom_demand
   utilities/save_load
   utilities/plot

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: References

   tutorials/api



