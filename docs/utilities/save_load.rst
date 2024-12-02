Save and Load Controllers
=========================

It is possible to save the trained controller to disk. This can be done using either Python’s pickle utility or controller's `save` and `load` method for SingleSourcingNeuralController and DualSourcingNeuralController.

.. code-block:: python
    
    import pickle
    

    from idinn.single_controller import SingleSourcingNeuralController

    # Save the model
    single_neural_controller.save("optimal_single_neural_controller.pt")
    # Load the model
    saved_single_controller = SingleSourcingNeuralController()
    saved_single_controller = saved_single_controller.load("optimal_single_neural_controller.pt")