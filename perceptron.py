import numpy as np
"""
https://machinelearningmastery.com/ \
    implement-perceptron-algorithm-scratch-python
The Perceptron algorithm is the simplest type of artificial neural network. The
artificial neuron receives one or more inputs (at neural dendrites) and sums
them to produce an output (or activation, representing a neuron's action
potential which is transmitted along its axon).

The perceptron is a model of a single neuron that can be used for two-class
classification problems and provides the foundation for later developing much
larger networks.
"""


class Perceptron:
    data = None
    labels = None
    weights = None

    def __init__(self, data, labels, activation_func):
        """
        Init function.
        """
        self.data = data
        self.labels = labels
        self.activation_func = activation_func
        self._init_weights()

    def _init_weights(self):
        """
        Function to initialize weights.

        The weight list will be initialized with the number of inputs plus an
        extra weight for the bias. To understand the bias a bit better, it is
        somehow similar to the constant b of a linear function:

        y = ax + b

        It allows you to move the line up and down to fit the prediction with
        the data better. Without b, the line always goes through the origin (0,
        0) and you may get a poorer fit. Also, multiplying a bias by a weight,
        you can shift it by an arbitrary amount.

        TODO: explain why we need random initialization of weights
        https://machinelearningmastery.com/ \
                why-initialize-a-neural-network-with-random-weights/

        """
        # Get the number of inputs plus extra one for bias
        weights_number = self.data.shape[1] + 1

        # Create random weights for main inputs and bias
        self.weights = np.random.randn(weights_number)

    def _row_predict(self, row):
        # The 1st weight is reserved for the bias
        activation = self.weights[0]
        for i in range(row.size):
            activation += self.weights[i+1] * row[i]
        return self.activation_func(activation)

    def predict(self):
        print("Weights={0}".format(self.weights))
        for input_i in range(0, self.labels.size):
            row = self.data[input_i]
            prediction = self._row_predict(row)
            print("Expected={0}, Predicted={1}".format(
                self.labels[input_i], prediction))

    def train(self, l_rate, n_epoch, epoch_show=5):
        """
        Estimate perceptron weights using stochastic gradient descent It uses
        online learning: the learning mode where the model update is performed
        each time a single observation is received.  This is different to
        "batch learning", where the model update is performed after observing
        the entire training set
        """
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row, label in zip(self.data, self.labels):
                prediction = self._row_predict(row)
                error = label[0] - prediction
                sum_error += error**2
                """
                Update weights
                w_i(t+1) = w_i(t) + r*(d_j - y_j(t))*x_i_j

                w: weight
                r: learning rate
                d: desired output (label)
                y: perceptron output
                x: input
                """
                self.weights[0] += l_rate * error
                for i in range(row.size):
                    self.weights[i+1] += l_rate * error * row[i]
            # Print just every epoch_show
            if (epoch + 1) % epoch_show == 0:
                print(">epoch={0}, lrate={1:.2f}, error={2:.2f}, weights={3}".
                      format(epoch, l_rate, sum_error, self.weights))
