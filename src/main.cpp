
//#define ENABLE_CUDA  1
#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>




int main()
{
    // cmake .. -DENABLE_CUDA=ON 
    
    

    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(64, Activation::RELU);
    c.add_layer(64, Activation::RELU);
    c.add_layer(10,  Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise_random_weights(-0.1, 0.1);


    Dataset train = Dataset("../datasets/mnist_train.csv");
    train.normalize();
    train.one_hot_encode();
    
    Dataset test = Dataset("../datasets/mnist_test.csv");
    test.normalize();
    test.one_hot_encode();     

    c.fit(1, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.01 , 256);


    c.performance(train, "train");
    c.performance(test, "test");

    c.save_weights("test1.txt");

    


}
