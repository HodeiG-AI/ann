import numpy as np
from perceptron import Perceptron
from activation_functions import binary_step
from utils import plot_2d

LR = 0.1
EPOCHS = 50

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
    perceptron = Perceptron(data, labels, binary_step)
    perceptron.train(LR, EPOCHS)
    plot_2d(data, labels, perceptron.weights)
