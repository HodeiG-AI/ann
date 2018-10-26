import numpy as np
import matplotlib.pyplot as plt

# https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# inputs = dendrites
# outputs = axon
BIAS = 1
LR = 0.1
EPOCHS = 50
EPOCH_SHOW = 5


def predict(inputs, weights):
    activation = 0
    # Add bias
    inputs = np.insert(inputs, 0, BIAS)
    for i in range(0, inputs.size):
        activation += weights[i] * inputs[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate perceptron weights using stochastic gradient descent
# It uses online learning: the learning mode where the model
# update is performed each time a single observation is received.
# This is different to "batch learning", where the model update
# is performed after observing the entire training set
def train(data, labels, l_rate, n_epoch):
    # Get the number of inputs plus extra one for bias
    inputs_number = data.shape[1] + 1

    # Create random weights for main inputs and bias
    # https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
    weights = np.random.randn(inputs_number)
    for epoch in range(n_epoch):
        sum_error = 0.0
        for inputs, label in zip(data, labels):
            prediction = predict(inputs, weights)
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
            for i in range(0, inputs.size):
                weights[i] += l_rate * error * inputs[i]
        # Print just every 20 epochs
        if (epoch + 1) % EPOCH_SHOW == 0:
            print(">epoch={0}, lrate={1:.2f}, error={2:.2f}, weights={3}".
                  format(epoch, l_rate, sum_error, weights))
    return weights


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
    weights = train(data, labels, LR, EPOCHS)
    # https://stackoverflow.com/questions/31292393/how-do-you-draw-a-line-using-the-weight-vector-in-a-linear-perceptron
    plt.figure()
    plt.grid()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Input data')
    for inputs, target in zip(data, labels):
        plt.plot(inputs[0], inputs[1], 'ro' if (target[0] == 1) else 'bo')
    #  plt.scatter(data[:, 0], data[:, 1])
    for i in np.linspace(np.amin(data), np.amax(data)):
        # Plot input data
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
        intercept = -weights[0]/weights[2]
        # y = mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        plt.plot(i, y, 'ko')
    plt.show()
