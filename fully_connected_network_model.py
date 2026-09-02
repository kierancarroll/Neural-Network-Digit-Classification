from layers.layer import Layer
from layers.linear import Linear
from layers.sigmoid import Sigmoid
from layers.softmax_loss import SoftmaxLoss

class FullyConnectedNetwork(Layer):
    def __init__(self, num_input, neurons_per_layer):
        """
        Instantiate all your layers here!

        Input:
            num_input (int): the dimension of the input vector to the neural network
            neurons_per_layer (list of ints): a list of integers where each integer is the number of neurons in that layer
                (including the output layer)

        Don't forget to initialize self.parameters with all the layers' parameters!
        Note: The test cases will not test the exact contents of parameters (so the order of parameters in the list doesn't matter)
            However, make sure that all the parameters are in self.parameters, since the test cases will test if the model exhibits
            the correct behavior after being updated by an optimizer
        """
        super().__init__()
        layers = []
        parameters = []
        gradients = []
        num_layers = len(neurons_per_layer)


        for i in range(num_layers):
          if i == num_layers - 1: #Last Linear + SoftmaxLoss layer
            linear_layer = Linear(num_input, neurons_per_layer[i])
            layers.append(linear_layer)
            parameters.append(linear_layer.parameters)
            gradients.append(linear_layer.gradients)
            layers.append(SoftmaxLoss())

          else: #Linear + Sigmoid layer
            linear_layer = Linear(num_input, neurons_per_layer[i])
            layers.append(linear_layer)
            parameters.append(linear_layer.parameters)
            gradients.append(linear_layer.gradients)
            layers.append(Sigmoid())

          num_input = neurons_per_layer[i]

        self.layers = layers
        self.parameters = parameters
        self.gradients = gradients



    def forward(self, x, y=None):
        """
        Forward pass for the neural network

        Input:
            x (numpy array): A batch of inputs dimensions (B, num_input), where B is the batch size
            y (numpy array): A batch of one-hot vector representations of the true labels with dimensions (B, num_classes),
                where B is the batch size. This is None when not being used for training.

        Returns:
            Tuple (y_hat, loss):
                y_hat (numpy array): a batch of probability distributions for the output classes with dimensions (B, num_classes)
                loss (numpy array): a batch of losses with dimensions (B,). This is None if y=None.

        """
        layers = self.layers
        num_layers = len(layers)
        z = x
        a = x

        for i in range(num_layers):
          if i == num_layers -1: #SoftmaxLoss layer
            if y is not None:
              final_result = layers[i].forward(z,y)
              return final_result
            else:
              final_result = layers[i].forward(z)
              return final_result
          elif i % 2 == 0: #Linear layer
            z = layers[i].forward(a)
          else: #Sigmoid layer
            a = layers[i].forward(z)

    def backward(self):
        """
        Backward pass for the neural network

        Returns:
            numpy array: gradient of the loss w.r.t. input, with dimensions (B, 784) where B is the batch size
        """
        layers = self.layers
        num_layers = len(layers)

        for i in range(num_layers-1,-1,-1):
          if i == num_layers - 1: #Last SoftmaxLoss layer
            next_input_grad = layers[i].backward()
          elif i % 2 == 0:  #Linear layer
            next_input_grad = layers[i].backward(next_input_grad)
          else: #Sigmoid layer
            next_input_grad = layers[i].backward(next_input_grad)
        return next_input_grad


    def get_first_layer_weights(self):
        """
        Gets the weights of the first linear layer: this function will be used for visualizing the weights.

        Returns:
            numpy array: 128x784 array, weights of the first linear layer
        """

        return self.parameters[0][0]
