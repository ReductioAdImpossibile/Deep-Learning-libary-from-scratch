#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{

    // TODO :
    // conv layer
    // cuda
    
    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(256, Activation::RELU);
    c.add_layer(128, Activation::RELU);
    c.add_layer(10, Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise_random_weights();

    
    Dataset train = Dataset("../datasets/mnist_train.csv");
    Dataset test = Dataset("../datasets/mnist_test.csv");
 
    train.normalize();
    test.normalize();

    train.one_hot_encode();
    test.one_hot_encode();
    


    c.fit(1, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , 2);


    
    ADAM_Optimizer adam;
    adam.batch_size = 1;
    adam.lambda = 0;
    //c.fit(1 , train, adam); // epoch >= batch_size

    //Hyperparameter p;
    //c.fit(1, train, Optimizer::STOCHASTIC_GRADIENT_DESCENT, p);

    c.performance(train);
    c.performance(test);

    c.save_weights("test1.txt");
}