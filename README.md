# Neural-Network-in-C-from-scratch
Neural Network in pure C (no libraries except for the basic ones) for digit recognition on the MNIST character dataset. 

This code is my implementation of a very basic neural network in pure C.
The model only contains 1 matrix multiplication between the input and weights matricies, and then a matrix addition with the bias matrix.
The model then applies a Relu function, and then a Softmax function.
The gradients are then calculated manually, and then added to the weights and biases matricies.

All of the basic functions to perform calculations on arrays(matrix multiplication, matrix addition, etc.) had to be coded from scratch.

The only thing that was not coded from scratch is the mnist.h file that allows me to load the data into an array. This was copied from Nuri Park - https://github.com/projectgalateia/mnist
model.c contains the model, which is really just a few operations inside the training loop, and the training loop.
functions.c and functions.h contain the functions I coded from scratch to perform the necessary basic operations.

Also the data is not present in this repository, but it is just the MNIST dataset.
