import numpy as np
from layers.layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        """
        The sigmoid activation function as no parameters or configurations. Nothing to do here!
        """
        super().__init__()

    def forward(self, z):
        """
        Computes the sigmoid for z

        Input:
            z (numpy array): input to the sigmoid, with dimensions (B, M), where B is the batch size

        Returns:
            numpy array: output of the sigmoid, with dimensions (B, M), where B is the batch size
        """
        a = 1/(1+(np.exp((-1*z))))
        self.forward_output = a
        return a

    def backward(self, gradient_loss_wrt_output):
        """
        Computes gradient of the input of the forward pass, given the gradient of loss w.r.t. the output
        Note - in the forward pass, you will need to store some values in the object to implement the backward pass efficiently
        Points will be manually deducted if you unnecessarily recompute  values in the backward pass.

        Input:
            grad_wrt_output (numpy array): gradient of the loss w.r.t. output, with dimensions (B, M), where B is the batch size

        Returns:
            numpy array: gradient of the input w.r.t. loss, with dimensions (B, M), where B is the batch size
        """
        gradient_loss_wrt_input = gradient_loss_wrt_output * self.forward_output * (1 - self.forward_output)
        return gradient_loss_wrt_input
