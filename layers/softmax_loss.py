import numpy as np
from layers.layer import Layer

class SoftmaxLoss(Layer):
    def __init__(self):
        """
        The softmax activation function and loss function have no parameters or configurations. Nothing to do here!
        """
        super().__init__()

    def forward(self, z, y=None):
        """
        Forward pass for SoftmaxLoss
        There are two modes to this function - training and evaluation (when the y argument is None)
            if y != None - take the softmax of z and compute the cross entropy loss (for each element of the batch)
                and return it. That is, return the loss(y, Softmax(z))
            else - Take the softmax of z and return the result. That is, return (Softmax(z), None)

        This is because when we train, we are interested in the losses of the model, but when evaluating or using our model, we want the layer
            to output predictions (i.e. the probability distributions for the output classes)

        Input:
            z (numpy array): The output logits from the neural network with dimensions (B, M), where B is the batch size
            y (numpy array): A batch of one-hot vector representations of the true labels with dimensions (B, num_classes),
                where B is the batch size. This is None when not being used for training.

        Returns:
            Tuple (y_hat, loss):
                y_hat (numpy array): a batch of probability distributions for the output classes with dimensions (B, num_classes)
                loss (numpy array): a batch of losses with dimensions (B,). This is None if y=None.
        """
        cross_entropy_loss = 0
        if y is not None:
          z_exp = np.exp(z)
          y_hat = z_exp / np.sum(z_exp, axis = 1, keepdims = True)

          y_times_y_hat_logged = y * np.log(y_hat)
          cross_entropy_loss = -1 * np.sum(y_times_y_hat_logged, axis = 1)
          self.y_hat = y_hat
          self.y = y
          return (y_hat, cross_entropy_loss)
        else:
          z_exp = np.exp(z)
          y_hat = z_exp / np.sum(z_exp, axis = 1, keepdims = True)
          return (y_hat, None)


    def backward(self):
        """
        Backward pass for softmax loss
        Note - in the forward pass, you will need to store some values in the object to implement the backward pass efficiently.
        Points will be manually deducted if you unnecessarily recompute values in the backward pass.
        The backward function should never be called outside of training

        Returns:
            numpy array: gradient of loss w.r.t. the input z
        """
        gradient_softmaxLoss = self.y_hat - self.y
        return gradient_softmaxLoss