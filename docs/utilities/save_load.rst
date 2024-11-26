Save and Load Controllers
=========================

It is also a good idea to save the trained neural network controller for future use. This can be done using the `save` method. The `load` method allows one to load a previously saved controller.

.. code-block:: python

    # Save the model
    single_controller.save("optimal_single_sourcing_controller.pt")
    # Load the model
    single_controller_loaded = SingleSourcingNeuralController(
        hidden_layers=[2], activation=torch.nn.CELU(alpha=1)
    )
    single_controller_loaded.load("optimal_single_sourcing_controller.pt")