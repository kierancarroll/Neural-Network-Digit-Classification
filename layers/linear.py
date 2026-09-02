import numpy as np
from layers.layer import Layer
from utils import initialize_random_array

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        """
        Initializes the parameters for the model. Note that the parameters for a linear layer are the weights matrix and
        the bias vector

        Input:
            in_dim (int): the dimension of the input vector to the linear layer
            out_dim (int): the dimension of the output vector to the linear layer

        IMPORTANT -- use initialize_random_array from utils.py to randomly initialize your arrays.
                     You need to do this to pass the autograder.
        """
        super().__init__()
        W = initialize_random_array((out_dim, in_dim)) # Initialize this in your solution!
        b = initialize_random_array((out_dim,)) # Initialize this in your solution!
        # Note: b should be a one-dimensional array (not 2 dimensional!)

        # Scale initial parameters by sqrt(1/in_dim) to improve training
        W *= np.sqrt(1/in_dim)
        b *= np.sqrt(1/in_dim)

        # The parameter attribute is initialized here. Use this attribute to access the parameters in
        # your forward and backward function implementations
        self.parameters = [W, b]
        self.gradients = [None, None] # To be set in the backwards pass

    def forward(self, x):
        """
        Forward pass for the linear layer

        Input:
            x (numpy array): Input to the linear layer with dimensions (B, in_dim), where B is the batch size

        Returns:
        numpy array: Output of the linear layer with dimensions (B, out_dim), where B is the batch size
        """
        # W and b initialized for you here. Use this for your forward implementation!
        W = self.parameters[0]
        b = self.parameters[1]

        self.x = x
        z = x @ W.T + b

        return z

    def backward(self, gradient_loss_wrt_output):
        """
        Backward pass for the linear layer and stored gradients of the loss w.r.t parameters
        Note - in the forward pass, you may need to store some values in the object to implement the backward pass

        Input:
            grad_wrt_output (numpy array): gradient of the loss w.r.t. output, with dimensions (B, out_dim), where B is the batch size

        Returns:
            numpy array: gradient of the loss w.r.t. input, with dimensions (B, in_dim) where B is the batch size

        Also:
        Update the gradients of the parameters in self.parameters:
         - the gradient of the loss w.r.t. W should be stored in self.gradients[0]
         - the gradient of the loss w.r.t. b should be stored in self.gradients[1]
        You should store the AVERAGE of the gradients across the batch.
        """
        W = self.parameters[0]
        b = self.parameters[1]

        gradient_loss_wrt_input = gradient_loss_wrt_output @ W
        gradient_loss_wrt_W = gradient_loss_wrt_output.T @ self.x
        gradient_loss_wrt_b = np.sum(gradient_loss_wrt_output, axis = 0)

        self.gradients[0] = gradient_loss_wrt_W / self.x.shape[0]
        self.gradients[1] = gradient_loss_wrt_b / self.x.shape[0]
        return gradient_loss_wrt_input
