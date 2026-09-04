import numpy as np
from utils import to_one_hot, clip_str, BatchedMNIST, plot_training_history, plot_confusion_matrix, plot_weight_heatmaps
from fully_connected_network_model import FullyConnectedNetwork
from optimizers.MBSGD_optimizer import SGD
from sklearn.metrics import confusion_matrix

def evaluate(batches, neural_network):
    """
        Evaluates the neural network for the given batches of data and returns the average loss and accuracy

    Input:
        batches (list of tuples): list of tuples where each tuple contains a batch of images and the corrsponding batch of labels
        neural_network (Layer): Neural network model to evaluate

    Returns:
        float: Average losses across all points in all batches
        float: Classification accuracy rate across all points in all batches
    """
    sum_loss = 0
    num_correct = 0
    num_points = 0

    for x_image, y_labels in batches:
        current_batch_size = x_image.shape[0]
        num_points += current_batch_size

        # The input batch and one-hot-vector labels
        x = x_image.reshape(current_batch_size, 784)
        y = to_one_hot(y_labels, 10)

        batch_y_hat, batch_loss = neural_network.forward(x,y)
        for i in range(batch_y_hat.shape[0]):
          predicted_class = np.argmax(batch_y_hat[i,:])
          if y[i][predicted_class] == 1: #YAY!
            num_correct += 1
        sum_loss += np.sum(batch_loss)


    avg_loss = sum_loss / num_points
    accuracy = num_correct / num_points

    return avg_loss, accuracy

def train_NN(
        neural_network, optimizer, num_epochs,
        batched_dataset_training, batched_dataset_validation,
        verbose=True,
    ):
    """
    Trains the neural network and records training metrics

    Input:
        neural_network (Layer): untrained neural network. This network will be trained in-place
        optimizer (Optimizer): Optimizer for neural_network
        num_epochs (int): number of epochs to train for
        batched_dataset_training (BatchedMNIST): training dataset
        batched_dataset_validation (BatchedMNIST): validation dataset
        verbose (bool): If True, print training progress

    Returns:
        float list: training losses - array of average training losses for each epoch
        float list: validation losses - array of average validation losses for each epoch
        float list: training accuracies - array of average training accuracy
        float list: validation accuracies - array of average validation accuracy
    """

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # Trainining loop

    for epoch_num in range(1, num_epochs+1):
        # Shuffle your training data after each epoch
        batched_dataset_training.shuffle()

        # Training
        for x_image, y_labels in batched_dataset_training:
            current_batch_size = x_image.shape[0]
            x = x_image.reshape(current_batch_size, 784)
            y = to_one_hot(y_labels, 10)

            neural_network.forward(x,y)
            grad_loss_wrt_x = neural_network.backward()
            optimizer.step()

        # Evaluation
        training_loss, training_accuracy = evaluate(batched_dataset_training, neural_network)
        validation_loss, validation_accuracy = evaluate(batched_dataset_validation, neural_network)

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        if verbose:
            print(
                f"Epoch {clip_str(epoch_num, 3)} | "
                f"Training loss {clip_str(training_losses[-1], 6)} | "
                f"Validation loss {clip_str(validation_losses[-1], 6)} | "
                f"Training accuracy {clip_str(training_accuracies[-1], 6)} | "
                f"Validation accuracy {clip_str(validation_accuracies[-1], 6)}"
            )

    return training_losses, validation_losses, training_accuracies, validation_accuracies

def process_confusion_matrix(net, batches):
    true_outputs = []
    predicted_outputs = []
    for image_batch, label_batch in batches:
        true_outputs += label_batch.tolist()

        y_hat, _ = net.forward(image_batch.reshape((image_batch.shape[0], 784)))
        predicted_outputs += np.argmax(y_hat, axis=1).tolist()

    cm = confusion_matrix(true_outputs, predicted_outputs)
    plot_confusion_matrix(cm)

batch_size = 64
train_data = BatchedMNIST(dataset="training", batch_size=batch_size, randomize=True, max_points=30000)
val_data = BatchedMNIST(dataset="validation", batch_size=batch_size, randomize=False, max_points=5000)

net = FullyConnectedNetwork(784, [64, 10])
first_layer_weights_untrained = net.get_first_layer_weights().copy() # For visualizing weights 
learning_rate = 1.0
optim = SGD(net, learning_rate)
num_epochs = 20

TL, VL, TA, VA = train_NN(net, optim, num_epochs, train_data, val_data)
first_layer_weights_trained = net.get_first_layer_weights().copy() # For visualizing weights 

plot_training_history(TL, VL, TA, VA)
plot_weight_heatmaps(first_layer_weights_untrained, "Untrained Weights")
plot_weight_heatmaps(first_layer_weights_trained, "Trained Weights")

process_confusion_matrix(net, val_data)