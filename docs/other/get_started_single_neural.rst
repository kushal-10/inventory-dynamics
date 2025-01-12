Solve Single-Sourcing Problems Using Neural Networks
====================================================

The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize inventory-related costs, given stochastic demand. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost :math:`h`. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost :math:`b`. This problem can be addressed using `idinn`. As illustrated in the :doc:`/get_started/get_started` section, we first initialize the sourcing model and its associated neural network controller. Subsequently, we train the neural network controller using data generated from the sourcing model. Finally, we use the trained neural network controller to compute optimal order quantities, which depend on the state of the system.

Initialization
--------------

We use the `SingleSourcingModel` class to initialize a single-sourcing model. The single-sourcing model considered in this example has a lead time of 0, i.e., the order arrives immediately after it is placed, and initial inventory of 10. The holding cost, :math:`h`, and the shortage cost, :math:`b`, are 5 and 495, respectively. The demand is generated from a discrete uniform distribution with support :math:`[1, 4]`. We use a batch size of 32 for training the neural network, i.e. the sourcing model generate 32 samples simultaneously.

.. code-block:: python
    
   import torch
   from idinn.sourcing_model import SingleSourcingModel
   from idinn.controller import SingleSourcingNeuralController
   from idinn.demand import UniformDemand

   single_sourcing_model = SingleSourcingModel(
       lead_time=0,
       holding_cost=5,
       shortage_cost=495,
       batch_size=32,
       init_inventory=10,
       demand_generator=UniformDemand(low=1, high=4),
    )

The cost at period :math:`t`, :math:`c_t`, is

.. math::

   c_t = h \max(0, I_t) + b \max(0, - I_t)\,,

where :math:`I_t` is the inventory level at the end of period :math:`t`. The higher the holding cost, the more costly it is to keep the inventory postive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The joint holding and stockout cost across all periods can be can be calculated using the `get_total_cost()` method of the sourcing model.

.. code-block:: python
    
   single_sourcing_model.get_total_cost()

In this example, this function should return 50 for each sample since the initial inventory is 10 and the holding cost is 5. We have 32 samples in this case, as we specified the batch size to be 32.

For single-sourcing problems, we use the neural network controller of the `SingleSourcingNeuralController` class. For illustration purposes, we employ a simple neural network with 1 hidden layer and 2 neurons. The activation function is `torch.nn.CELU(alpha=1)`.

.. code-block:: python

    single_controller = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )

Training
--------

Although the neural network controller has not been trained yet, we can still compute the total cost associated with its order policy. To do so, we integrate it with our previously specified sourcing model and run simulations for 100 periods.

.. code-block:: python
    
    single_controller.get_total_cost(
        sourcing_model=single_sourcing_model,
        sourcing_periods=100
        )

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train()` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the `seed` parameter.

.. code-block:: python

    from torch.utils.tensorboard import SummaryWriter

    single_controller.fit(
        sourcing_model=single_sourcing_model,
        sourcing_periods=50,
        validation_sourcing_periods=1000,
        epochs=5000,
        seed=1,
        tensorboard_writer=SummaryWriter(comment="single")
    )

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

.. code-block:: python

    single_controller.get_total_cost(
        sourcing_model=single_sourcing_model,
        sourcing_periods=100
        )

Plotting and Order Calculation
------------------------------------------

We can inspect how the controller performs in the specified sourcing environment by plotting the inventory and order histories.

.. code-block:: python

    # Simulate and plot the results
    single_controller.plot(sourcing_model=single_sourcing_model, sourcing_periods=100)

.. image:: ../_static/single_sourcing_output.png
   :alt: Output of the single sourcing model and controller
   :align: center

Then we can calculate optimal orders using the trained model.

.. code-block:: python
    # Calculate the optimal order quantity for applications
    single_controller.forward(current_inventory=10, past_orders=[1, 5])

Save and Load the Model
-----------------------

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows one to load a previously saved controller.

.. code-block:: python

    # Save the model
    single_controller.save("optimal_single_sourcing_controller.pt")
    # Load the model
    single_controller_loaded = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )
    single_controller_loaded.load("optimal_single_sourcing_controller.pt")