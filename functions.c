// #define USE_MNIST_LOADER
// #define MNIST_DOUBLE

// #include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <time.h>

#include "functions.h"

// I COPIED THIS FUNCTION
// void writeToCSV(const char *filename, double loss) {
//     FILE *fp = fopen(filename, "a");
//     fseek(fp, 0, SEEK_END);
//     if (ftell(fp) == 0) {
//         fprintf(fp, "loss\n");
//     }

//     fprintf(fp, "%lf\n", loss);
//     fclose(fp);
// }

// efficient
void print2dDoubleArray(double** array, int numRows, int numCols){
    printf("[");
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            printf("%lf", array[i][j]);
            if (j < numCols-1){
                printf(", ");
            }
        }
        if (i < numRows-1){
            printf("\n");
        }
    }
    printf("]\n");
}
// efficient
double meanSquaredError(double** target, double** output, int numRows, int numCols){
    double loss = 0.0;
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            loss += pow((target[i][j] - output[i][j]), 2);
        }
    }
    loss /= numRows*numCols; // may need chaged if I add batches...
    return loss;
}
// efficient
double generateRandomNumber(){
    return ((double)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1;
}
// not used?
// double** generateDoubleValue2dArray(int numRows, int numCols, double value){
//     double** array = (double**)malloc(numRows * sizeof(double*));
//     for (int i = 0; i < numRows; i++) {
//         array[i] = (double*)malloc(numCols * sizeof(double));
//         for (int j = 0; j < numCols; j++){
//             array[i][j] = value;
//         }
//     }
//     return array;
// }
// efficient
double** generateRandom2dArray(int numRows, int numCols){
    double** array = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; i++) {
        array[i] = (double*)malloc(numCols * sizeof(double));
        for (int j = 0; j < numCols; j++){
            array[i][j] = generateRandomNumber(); // replace with actual random standard distrubtion weights
        }
    }
    return array;
}
// efficient
double** generateZero2dArray(int numRows, int numCols){
    double** array = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; i++) {
        array[i] = (double*)malloc(numCols * sizeof(double));
        for (int j = 0; j < numCols; j++){
            array[i][j] = 0.0;
        }
    }
    return array;
}
// should free array? free original array if it is no longer being used
// make sure that I am always setting the array that gets inputted into the function to be equal to the function
// free
double** zero(double** array, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[i][j] = 0.0;
        }
    }
    // free(array);
    return result;
}// this approach where I make an array and return it may not be effecient
// free
double** transpose(double** array, int numRows, int numCols){
    double** result = generateZero2dArray(numCols, numRows);
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[j][i] = array[i][j];
        }
    }
    // free(array);
    return result;
}

// double** getBatch(mnist_data data, int dataNumRows, int dataNumCols, int inputNumRows, int inputNumCols){
//     double** input = generateZero2dArray(inputNumRows, inputNumCols);
//     for (int i = 0; i < dataNumRows; i++){
//         for (int j = 0; j < dataNumCols; j++){
//             input[0][i*dataNumCols + j] = data.data[i][j];
//         }
//     }
//     return input;
// }
// efficient
double** oneHotEncode(int trainLabel, int numRows, int numCols){ // need to change for batched input
    double** result = generateZero2dArray(numRows, numCols);
    result[0][trainLabel] = 1.0;
    return result;
}
// cannot free(matrix) because I still need logits...
double** softmax(double** matrix, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);
    for (int i = 0; i < numRows; i++){
        double rowSum = 0.0;
        for (int j = 0; j < numCols; j++){
            // result[i][j] = exp(matrix[i][j]); // can replace exp() with my own exp function of the taylor series expansion
            rowSum += exp(matrix[i][j]);
        }
        for (int j = 0; j < numCols; j++){
            result[i][j] = exp(matrix[i][j]) / rowSum;
        }
    }
    return result;
}
// free
double** matmul(double** mat1, double** mat2, int numRows1, int numCols1, int numRows2, int numCols2){
    // dynamically allocate 2d result array
    double** result = generateZero2dArray(numRows1, numCols2);
    // matrix multiplication
    for (int i = 0; i < numRows1; i++){
        for (int j = 0; j < numCols2; j++){ // rows from mat1 add to columns from mat2 // over and down
            result[i][j] = 0;
            for (int k = 0; k < numCols1; k++){ // numCols1 == numRows2
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}
// free
double** matAdd(double** mat1, double** mat2, int numRows, int numCols){ // must be same shape
    double** result = generateZero2dArray(numRows, numCols);

    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;
}
// free
double** relu(double** mat1, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);

    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            if (mat1[i][j] < 0){
                result[i][j] = 0;
            }
            else{
                result[i][j] = mat1[i][j];
            }
        }
    }
    return result;
}
