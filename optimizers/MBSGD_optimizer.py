from optimizers.base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, model, learning_rate):
        """
        Stores parameters in the optimizer and stores the learning rate

        Input:
            model (Layer): the model to be optimized
            learning_rate (float): learning rate of the SGD optimizer
        """

        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        """
        Updates parameters in self.parameters using the learning rate and gradients.
        Use self.model.get_parameters() and self.model.get_gradients() to access the parameters of the model
        and gradients computed by the backward pass. (Assume the step function is only called after the backward
        pass is run)

        IMPORTANT: Use `+=` and `-=` to update parameters. This would update the numpy arrays in-place.
        """
        #this is one epoch
        parameter_list = self.model.get_parameters()
        gradient_list = self.model.get_gradients()

        for i in range(len(parameter_list)):
          if len(self.model.parameters[i]) == 2:
            self.model.parameters[i][0] -= self.learning_rate * gradient_list[i][0]
            self.model.parameters[i][1] -= self.learning_rate * gradient_list[i][1]
          else:
            self.model.parameters[i] -= self.learning_rate * gradient_list[i]
