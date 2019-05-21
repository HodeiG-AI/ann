import numpy as np
import matplotlib.pyplot as plt


def plot_2d(data, labels, weights):
    """
    Function to plot a line using the weight vector in a linear perceptron
    https://stackoverflow.com/questions/31292393/ \
    how-do-you-draw-a-line-using-the-weight-vector-in-a-linear-perceptron \
    """
    fig = plt.figure()
    sp = fig.add_subplot(111)
    sp.grid()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Input data')
    for inputs, target in zip(data, labels):
        sp.plot(inputs[0], inputs[1], 'ro' if (target[0] == 1) else 'bo')
    #  plt.scatter(data[:, 0], data[:, 1])
    for i in np.linspace(np.amin(data), np.amax(data)):
        # Plot input data
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
        intercept = -weights[0]/weights[2]
        # y = mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        sp.plot(i, y, 'ko')
    # https://stackoverflow.com/questions/46880493/ \
    # matplotlib-pyplot-figure-show-freezes-instantly
    root = fig.canvas._tkcanvas.winfo_toplevel()  # Get tkinter root
    fig.show()
    root.mainloop()
