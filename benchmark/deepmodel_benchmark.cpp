#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

const std::string testset_path = "../datasets/mnist_test.csv";
const std::string trainset_path = "../datasets/mnist_train.csv";


void benchmark(const size_t batch_size, const size_t epochs, NeuralNetwork nn, Dataset &train, Dataset& test)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    nn.fit(epochs, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , batch_size);
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = end - start;
    std::cout << "[batch_size = " << batch_size << ", epochs = " << epochs << " => time : " << time.count() << "s , accuracy : "  << nn.accuracy(test) * 100 << "% ]" << std::endl;
  
    
}

void benchmark_adam(const size_t batch_size, const size_t epochs, NeuralNetwork nn, Dataset &train, Dataset& test)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = batch_size;
    adam.lambda = 10e-4;

    nn.fit(epochs, train, adam);
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = end - start;
    std::cout << "[batch_size = " << batch_size << ", epochs = " << epochs << " => time : " << time.count() << "s , accuracy : "  << nn.accuracy(test) * 100 << "% ]" << std::endl;
  
    
}

int main()
{

    Dataset train = Dataset(trainset_path);
    Dataset test = Dataset(testset_path);
    
    train.normalize();
    test.normalize();

    train.one_hot_encode();
    test.one_hot_encode();     


    NeuralNetwork nn;
    nn.disable_print();

    nn.configure_input_layer(784);
    nn.add_layer(128, Activation::RELU);
    nn.add_layer(128, Activation::RELU);
    nn.add_layer(10,  Activation::SOFTMAX);
    nn.configure_loss_function(Loss::CROSS_ENTROPY);
    
    nn.initalise_he_weights();
    

    std::cout << "----------------------------" << std::endl;
    benchmark(1, 1, nn, train, test);
    benchmark(2, 1, nn, train, test);
    benchmark(4, 1, nn, train, test);
    benchmark(8, 1, nn, train, test);
    benchmark(16, 1, nn, train, test);
    benchmark(32, 1, nn, train, test);
    benchmark(64, 1, nn, train, test);

    std::cout << "----------------------------" << std::endl;
    benchmark_adam(1, 1, nn, train, test);
    benchmark_adam(2, 1, nn, train, test);
    benchmark_adam(4, 1, nn, train, test);
    benchmark_adam(8, 1, nn, train, test);
    benchmark_adam(16, 1, nn, train, test);
    benchmark_adam(32, 1, nn, train, test);
    benchmark_adam(64, 1, nn, train, test);




}
