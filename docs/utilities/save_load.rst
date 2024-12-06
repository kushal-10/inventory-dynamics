Save and Load Controllers
=========================

Trained controllers can be saved to disk for later use. For the :class:`SingleSourcingNeuralController` and :class:`DualSourcingNeuralController`, this can be achieved using their `save` and `load` methods. For other controllers, Python's pickle utility can be used instead.

.. code-block:: python
    
    import pickle

    from idinn.single_controller import SingleSourcingNeuralController

    # Save the model
    single_neural_controller.save("optimal_single_neural_controller.pt")
    # Load the model
    saved_single_controller = SingleSourcingNeuralController()
    saved_single_controller = saved_single_controller.load("optimal_single_neural_controller.pt")