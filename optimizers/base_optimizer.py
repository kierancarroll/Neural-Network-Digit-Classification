class Optimizer:
    def __init__(self, model):
        """
        In the constructor, the model is passed in.
        Additional configurations (such as the learning rate) can be passed in as well.

        Input:
            model (Layer): the model to be optimized
        """
        self.model = model

    def __step__(self):
        """
        When this is called, the new parameter values are computed from the configurations from the
        constructor and the gradients of the parameters. The parameters are then updated in-place with
        the new parameter values.

        The parameters' values and gradients can be accessed using self.model.get_parameters() and self.model.get_gradients()
        """
        pass