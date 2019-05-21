"""
Defining activation functions (af_)
===========================
The activation function of a node defines the output of that node given an
input or set of inputs. The activation function is usually an abstraction
representing the rate of action potential firing in the cell. In its simplest
form, this function is binary, that is, either the neuron is firing or not.

https://en.wikipedia.org/wiki/Activation_function
    - Identity: f(x) = x                    Range: (-inf, inf)
                           / 0 for x < 0
    - Binary step: f(x) = /                 Range: {0, 1}
                          \
                           \ 1 for x>= 0
    - TanH: f(x): tanh(x)                   Range: (-1, 1)
                   / 0 for x < 0
    - ReLu f(x) = /                         Range: [0, inf)
                  \
                   \ x for x>= 0

"""


def binary_step(x):
    """
    Binary step activation function.

                           / 0 for x < 0
    - Binary step: f(x) = /                 Range: {0, 1}
                          \
                           \ 1 for x>= 0
    """
    return 1.0 if x >= 0.0 else 0.0
