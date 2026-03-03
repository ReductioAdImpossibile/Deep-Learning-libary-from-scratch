#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{


    // TODO : model und classificator vereinen
    // classificator umbennen 
    // dataset fertig, error checking nach fit verschieben!
    // Adam optimizer fit
    // fit fertig machen -> Batch GD, mini Batch GD
    // save + load
    // conv layer
    // cuda
    
    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(128, Activation::RELU);
    c.add_layer(64, Activation::RELU);
    c.add_layer(10, Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise();

    Dataset train = c.load_csv("../datasets/mnist_train.csv");
    Dataset test = c.load_csv("../datasets/mnist_test.csv");

    train.normalize();
    test.normalize();


    c.fit(10*60000, train, Optimizer::STOCHASTIC_GRADIENT_DESCENT, 0.001);

    c.performance(train);
    c.performance(test);

    //FUNKTIONIERT.

    /*
    Dataset ds;
    ds.expected.resize(4);
    ds.input.resize(4);

    ds.input[0] = Matrix(4, 1, {0,0,1,0});
    ds.expected[0] = Matrix(4, 1, {0,0,1,0 });

    ds.input[1] = Matrix(4, 1, {0,0,0,1});
    ds.expected[1] = Matrix(4, 1, {0,0,0,1 });

    ds.input[2] = Matrix(4, 1, {0,1,0,0});
    ds.expected[2] = Matrix(4, 1, {0,1,0,0 });

    ds.input[3] = Matrix(4, 1, {1,0,0,0});
    ds.expected[3] = Matrix(4, 1, {1,0,0,0 });


    Classificator c;
    c.configure_input_layer(4);
    c.add_layer(4, Activation::SIGMOID);
    c.configure_loss_function(Loss::QUADRATIC);

    c.initalise();
    c.fit(2000, Optimizer::STOCHASTIC_GRADIENT_DESCENT, ds);

    //c.run(ds.input[1]).print();
    //ds.expected[1].print();
    c.performance(ds);
    */




}