class Layer:
    def __init__(self):
        """
        For initializing parameters/configurations for the model layer
        """
        self.parameters = []
        self.gradients = []
        pass

    def forward(self, input):
        """
        Forward pass for the layer, returns the output of the forward pass
        The input can be overridden
        """
        pass

    def backward(self, grad_wrt_output):
        """
        Backward pass for the layer - calculates/records gradients for the input and the parameters
        Returns the gradients of the input
        """
        pass

    def get_parameters(self):
        """
        Gets the list of parameters.
        """
        return self.parameters

    def get_gradients(self):
        """
        Gets the list of gradients.
        """
        return self.gradients
