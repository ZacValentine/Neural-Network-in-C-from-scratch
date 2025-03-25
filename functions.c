#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "functions.h"


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

double meanSquaredError(double** target, double** output, int numRows, int numCols){
    double loss = 0.0;
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            loss += pow((target[i][j] - output[i][j]), 2);
        }
    }
    loss /= numRows*numCols;
    return loss;
}

double generateRandomNumber(){
    return ((double)rand() / RAND_MAX * 2.0f - 1.0f) * 0.1;
}

double** generateRandom2dArray(int numRows, int numCols){
    double** array = (double**)malloc(numRows * sizeof(double*));
    
    for (int i = 0; i < numRows; i++) {
        array[i] = (double*)malloc(numCols * sizeof(double));
        for (int j = 0; j < numCols; j++){
            array[i][j] = generateRandomNumber();
        }
    }
    return array;
}

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

double** zero(double** array, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);
    
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[i][j] = 0.0;
        }
    }

    return result;
}

double** transpose(double** array, int numRows, int numCols){
    double** result = generateZero2dArray(numCols, numRows);
    
    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[j][i] = array[i][j];
        }
    }

    return result;
}

double** oneHotEncode(int trainLabel, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);
    
    result[0][trainLabel] = 1.0;
    return result;
}

double** softmax(double** matrix, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);
    
    for (int i = 0; i < numRows; i++){
        double rowSum = 0.0;
        for (int j = 0; j < numCols; j++){
            rowSum += exp(matrix[i][j]);
        }
        for (int j = 0; j < numCols; j++){
            result[i][j] = exp(matrix[i][j]) / rowSum;
        }
    }
    return result;
}

double** matmul(double** mat1, double** mat2, int numRows1, int numCols1, int numRows2, int numCols2){
    double** result = generateZero2dArray(numRows1, numCols2);
    
    for (int i = 0; i < numRows1; i++){
        for (int j = 0; j < numCols2; j++){
            result[i][j] = 0;
            for (int k = 0; k < numCols1; k++){
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

double** matAdd(double** mat1, double** mat2, int numRows, int numCols){
    double** result = generateZero2dArray(numRows, numCols);

    for (int i = 0; i < numRows; i++){
        for (int j = 0; j < numCols; j++){
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;
}

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
