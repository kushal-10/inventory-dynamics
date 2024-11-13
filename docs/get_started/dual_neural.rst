Solve Dual-Sourcing Problems Using Neural Networks
==================================================

Initialization
--------------

To address dual-sourcing problems, we employ two main classes: (i) `DualSourcingModel` and (ii) `DualSourcingNeuralController`, responsible for setting up the sourcing model and its corresponding controller. In this tutorial, we examine a dual-sourcing model characterized by the following parameters: the regular order lead time is 2; the expedited order lead time is 0; the regular order cost, :math:`c^r`, is 0; the expedited order cost, :math:`c^e`, is 20; and the initial inventory is 6. Additionally, the holding cost, :math:`h`, and the shortage cost, :math:`b`, are 5 and 495, respectively. Demand is generated from a discrete uniform distribution with support :math:`[1, 4]`. In this example, we use a batch size of 256. 

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

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = c^r q^r_t + c^e q^e_t + h \max(0, I_t) + b \max(0, - I_t)\,,

where :math:`I_t` is the inventory level at the end of period :math:`t`, :math:`q^r_t` is the regular order placed at period :math:`t`, and :math:`q^e_t` is the expedited order placed at period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory positive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The higher the regular and expedited order costs, the more costly it is to place the respective orders. The joint holding and stockout cost across all periods can be can be calculated using the `get_total_cost()` method of the sourcing model.

.. code-block:: python
    
   dual_sourcing_model.get_total_cost(regular_q=0, expedited_q=0)

In this example, this function should return 30 for each sample since the initial inventory is 6, the holding cost is 5, and there is neither a regular nor expedited order. We have 256 samples in this case, as we specified a batch size of 256.

For dual-sourcing problems, we initialize the neural network controller using the `DualSourcingNeuralController` class. In this tutorial, we employ a simple neural network with 6 hidden layers. The numbers of neurons in each layer are 128, 64, 32, 16, 8, and 4, respectively. The activation function is `torch.nn.CELU(alpha=1)`.

.. code-block:: python

    dual_controller = DualSourcingNeuralController(
        hidden_layers=[128, 64, 32, 16, 8, 4],
        activation=torch.nn.CELU(alpha=1)
    )

Training
--------

Although the neural network controller has not been trained yet, we can still compute the total cost associated with its order policy. To do so, we integrate it with our previously specified sourcing model and run simulations for 100 periods.

.. code-block:: python

    dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train()` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the  `seed` parameter.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    dual_controller.fit(
        sourcing_model=dual_sourcing_model,
        sourcing_periods=100,
        validation_sourcing_periods=1000,
        epochs=2000,
        tensorboard_writer=SummaryWriter(comment="dual"),
        seed=4,
    )

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

.. code-block:: python
    
    dual_controller.get_total_cost(sourcing_model=dual_sourcing_model, sourcing_periods=100)

Plotting and Order Calculation
------------------------------------------

We can inspect how the controller performs in the specified sourcing environment by plotting the inventory and order histories.

.. code-block:: python

    # Simulate and plot the results
    dual_controller.plot(sourcing_model=dual_sourcing_model, sourcing_periods=100)

.. image:: ../_static/dual_sourcing_output.png
   :alt: Output of the dual sourcing model and controller
   :align: center

Then we can use the trained network to calculate near-optimal orders.

.. code-block:: python

    # Calculate the optimal order quantity for applications
    regular_q, expedited_q = dual_controller.forward(
        current_inventory=10,
        past_regular_orders=[1, 5],
        past_expedited_orders=[0, 0],
    )

Save and Load the Model
-----------------------

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save()` method. The `load()` method allows one to load a previously saved model.

.. code-block:: python

    # Save the model
    dual_controller.save("optimal_dual_sourcing_controller.pt")
    # Load the model
    dual_controller_loaded = DualSourcingNeuralController()
    dual_controller_loaded.load("optimal_dual_sourcing_controller.pt")