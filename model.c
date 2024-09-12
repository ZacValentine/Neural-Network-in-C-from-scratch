#include <stdio.h>
#include <stdlib.h>
#define USE_MNIST_LOADER // ONLY IMPORT/CODE THAT IS NOT MINE
#define MNIST_DOUBLE // ONLY IMPORT/CODE THAT IS NOT MINE
#include "mnist.h" // ONLY IMPORT/CODE THAT IS NOT MINE
#include "functions.h"

// gcc -o output model.c functions.c

// cant seem to get this one to work in the .h file
double** getBatch(mnist_data data, int dataNumRows, int dataNumCols, int inputNumRows, int inputNumCols){
    double** input = generateZero2dArray(inputNumRows, inputNumCols);
    for (int i = 0; i < dataNumRows; i++){
        for (int j = 0; j < dataNumCols; j++){
            input[0][i*dataNumCols + j] = data.data[i][j];
        }
    }
    return input;
}

// constants
// #define filename "saved/loss.csv" // file to save data to
#define mnistNumRows 28 // number of rows and rows in mnist 2d image
#define mnistNumCols 28
#define inputDim (mnistNumRows*mnistNumCols) // flattened 2d pixel array
#define outputDim 10 // num classes
#define batchSize 1 // for simplicity for now
#define numTrainImages 70006 // number of train images - replace with constant from the .h file

// parameter array shapes, used for calculations that involve iterating through all of a parameters rows/columns
// all shape nums: 1, 10, 768 // batchSize = 1, outputDim = 10, inputDim = 768
#define inputNumRows batchSize
#define inputNumCols inputDim
#define targetNumRows batchSize
#define targetNumCols outputDim
#define weights1NumRows inputDim
#define weights1NumCols outputDim
#define weightsOutputNumRows batchSize
#define weightsOutputNumCols outputDim
#define bias1NumRows 1
#define bias1NumCols outputDim
#define biasOutputNumRows batchSize
#define biasOutputNumCols outputDim
#define logitsNumRows batchSize
#define logitsNumCols outputDim
#define probsNumRows batchSize
#define probsNumCols outputDim

int main(){
    // setup data loading - ONLY CODE THAT IS NOT MINE
    mnist_data *data;
    unsigned int cnt;
    int ret;
    mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", &data, &cnt);
    printf("number of train images: %d\n", cnt);

    // random
    srand(7);

    // make stuct that contains the array, the shape of the array, and functions to modify them dynamically(appending, etc.)
    // make stuct that contains the model and uses the dynamic array struct for the parameters

    // tweakable parameters initialization
    double** weights1 = generateRandom2dArray(weights1NumRows, weights1NumCols);
    double** bias1 = generateRandom2dArray(bias1NumRows, bias1NumCols);

    // other array initialization
    double** input = generateZero2dArray(inputNumRows, inputNumCols);
    double** target = generateZero2dArray(targetNumRows, targetNumCols);

    double** weightsOutput = generateZero2dArray(weightsOutputNumRows, weightsOutputNumCols);
    double** biasOutput = generateZero2dArray(biasOutputNumRows, biasOutputNumCols);
    double** logits = generateZero2dArray(logitsNumRows, logitsNumCols);
    double** probs = generateZero2dArray(probsNumRows, probsNumCols);

    double** dprobs = generateZero2dArray(probsNumRows, probsNumCols);
    double** dlogits = generateZero2dArray(logitsNumRows, logitsNumCols);
    double** dbiasOutput = generateZero2dArray(biasOutputNumRows, biasOutputNumCols);
    double** dbias1 = generateZero2dArray(bias1NumRows, bias1NumCols);
    double** dweightsOutput = generateZero2dArray(weightsOutputNumRows, weightsOutputNumCols);
    double** dweights1 = generateZero2dArray(weights1NumRows, weights1NumCols);

    double loss = 0.0;
    double learningRate = -0.1;

    int numEpochs = 1;
    for (int epoch = 0; epoch < numEpochs; epoch++){
        for (int step = 0; step < numTrainImages; step++){
            // examine each function for effeciency, get better print statements + visualization + valdiation + etc., make it clean
            // forward pass
            input = getBatch(data[step], mnistNumRows, mnistNumCols, inputNumRows, inputNumCols);
            target = oneHotEncode(data[step].label, probsNumRows, probsNumCols);

            weightsOutput = matmul(input, weights1, inputNumRows, inputNumCols, weights1NumRows, weights1NumCols);
            biasOutput = matAdd(weightsOutput, bias1, inputNumRows, weights1NumCols);
            logits = relu(biasOutput, inputNumRows, weights1NumCols);

            probs = softmax(logits, inputNumRows, weights1NumCols);
            loss = meanSquaredError(target, probs, probsNumRows, probsNumCols);

            printf("%d: %lf\n", step, loss);
            printf("train_label: %d\n", data[step].label);
            printf("probs: ");
            print2dDoubleArray(probs, probsNumRows, probsNumCols);
    
            // backpropagation(calculate gradients manually)
            // how much does probs affect the loss through the meanSquaredError function... * dloss = how much it affects the loss
            for (int i = 0; i < probsNumRows; i++){
                for (int j = 0; j < probsNumCols; j++){
                    dprobs[i][j] = (-2.0/probsNumCols) * (target[i][j] - probs[i][j]);
                }
            }
            // how much does logits affect probs through the softmax function... * dprobs = how much it affects the loss
            for (int row = 0; row < logitsNumRows; row++){
                for (int i = 0; i < logitsNumCols; i++){
                    for (int j = 0; j < logitsNumCols; j++){
                        if (i == j){
                            dlogits[row][i] += (probs[row][i] * (1 - probs[row][i])) * dprobs[row][i];
                        }else{
                            dlogits[row][i] += (-probs[row][i] * probs[row][j]) * dprobs[row][j];
                        }
                    }
                }
            }
            // how much does biasOutput affect logits through relu... * dlogits = how much it affects the loss
            for (int i = 0; i < biasOutputNumRows; i++){
                for (int j = 0; j < biasOutputNumCols; j++){
                    if (logits[i][j] > 0){
                        dbiasOutput[i][j] = 1 * dlogits[i][j];
                    } else {
                        dbiasOutput[i][j] = 0 * dlogits[i][j];
                    }
                }
            }
            // how much does bias1 affect biasOutput through matAdd(weightsOutput, bias1)... * dbiasOutput = how much it affects the loss
            for (int i = 0; i < bias1NumRows; i++){
                for (int j = 0; j < bias1NumCols; j++){
                    dbias1[i][j] = 1 * dbiasOutput[i][j];
                }
            }
            // how much does weightsOutput affect biasOutput through matAdd(weightsOutput, bias1)... * dbiasOuptut = how much it affects the loss
            // some of these can be replaced with matmul or other operations
            for (int i = 0; i < weightsOutputNumRows; i++){
                for (int j = 0; j < weightsOutputNumCols; j++){
                    dweightsOutput[i][j] = 1 * dbiasOutput[i][j];
                }
            }
            // how much does weights1 affect weightsOutput through matmul(input, weights1)... * dweightsOutput = how much it affects the loss
            // matmul 768, 1 * 1, 10
            dweights1 = matmul(transpose(input, inputNumRows, inputNumCols), dweightsOutput, inputNumCols, inputNumRows, weightsOutputNumRows, weightsOutputNumCols);

            // printAll(), //printGradients()
            // // print gradients here

            // optimization(add gradients * learningRate to weights and biases)
            // weights1 
            for (int i = 0; i < weights1NumRows; i++){
                for (int j = 0; j < weights1NumCols; j++){
                    weights1[i][j] += dweights1[i][j] * learningRate;
                }
            }
            // bias1
            for (int i = 0; i < bias1NumRows; i++){
                for (int j = 0; j < bias1NumCols; j++){
                    bias1[i][j] += dbias1[i][j] * learningRate;
                }
            }

            // zero grad here - replace with function zeroGrad(params)
            dprobs = zero(dprobs, probsNumRows, probsNumCols);
            dlogits = zero(dlogits, logitsNumRows, logitsNumCols);
            dbiasOutput = zero(dbiasOutput, biasOutputNumRows, biasOutputNumCols);
            dbias1 = zero(dbias1, bias1NumRows, bias1NumCols);
            dweightsOutput = zero(dweightsOutput, weightsOutputNumRows, weightsOutputNumCols);
            dweights1 = zero(dweights1, weights1NumRows, weights1NumCols);

            // validation here
            //save to visualization file here
            // writeToCSV(filename, loss);
        }
    }
    return 0;
}