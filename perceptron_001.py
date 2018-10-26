import numpy as np
#  import matplotlib.pyplot as plt

# https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# inputs = dendrites
# outputs = axon
BIAS = 1


def predict(inputs, weights):
    # As the bias value will be
    activation = 0
    for i in range(0, inputs.size):
        activation += weights[i] * inputs[i]
    return 1.0 if activation >= 0.0 else 0.0


if __name__ == "__main__":
    # Load input data
    text = np.loadtxt('data/perceptron.txt')

    # Separate datapoints and labels
    data = text[:, :2]
    labels = text[:, 2].reshape((text.shape[0], 1))
    """
    (Pdb) data
    array([[0.38, 0.19],
           [0.17, 0.31],
           [0.29, 0.54],
           [0.89, 0.55],
           [0.78, 0.36]])
    (Pdb) labels
    array([[0.],
           [0.],
           [0.],
           [1.],
           [1.]])
    """
    # Get the number of inputs plus extra one for bias
    inputs_number = data.shape[1] + 1

    # Create random weights for main inputs and bias
    # https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
    weights = np.random.randn(inputs_number)

    for input_i in range(0, labels.size):
        inputs = np.insert(data[input_i], 0, BIAS)
        prediction = predict(inputs, weights)
        print("Expected={0}, Predicted={1}".format(
            labels[input_i], prediction))
