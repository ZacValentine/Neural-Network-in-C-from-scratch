#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// #define USE_MNIST_LOADER
// #define MNIST_DOUBLE

// #include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


void print2dDoubleArray(double** array, int numRows, int numCols);

double meanSquaredError(double** target, double** output, int numRows, int numCols);

double generateRandomNumber();

double** generateDoubleValue2dArray(int numRows, int numCols, double value);

double** generateRandom2dArray(int numRows, int numCols);

double** generateZero2dArray(int numRows, int numCols);

double** zero(double** array, int numRows, int numCols);

double** transpose(double** array, int numRows, int numCols);

// double** getBatch(mnist_data data, int dataNumRows, int dataNumCols, int inputNumRows, int inputNumCols);

double** oneHotEncode(int trainLabel, int numRows, int numCols);

double** softmax(double** matrix, int numRows, int numCols);

double** matmul(double** mat1, double** mat2, int numRows1, int numCols1, int numRows2, int numCols2);

double** matAdd(double** mat1, double** mat2, int numRows, int numCols);

double** relu(double** mat1, int numRows, int numCols);

#endif
