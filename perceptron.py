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
