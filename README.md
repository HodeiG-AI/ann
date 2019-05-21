# ann
Perceptron 001
==============
The 1st script will show how to develop a function that can make predictions.
We will build a perceptron that has 2 inputs (dendrites) and 1 output (axon).
For the prediction, the neuron will have 3 weights, the bias and a weight for
each input. To understand the bias a bit better, it is somehow similar to the
constant b of a linear function:

y = ax + b

It allows you to move the line up and down to fit the prediction with the data
better. Without b, the line always goes through the origin (0, 0) and you may
get a poorer fit. Also, multiplying a bias by a weight, you can shift it by an
arbitrary amount.

In our perceptron the activation function will look more like the below
function:

y = w0 + i1*w1 + i2*w2