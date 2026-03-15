#pragma once

#include "activation.h"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>


class Dataset
{

public:
    Matrix input;
    Matrix expected;
    
    Dataset();
    Dataset(const std::string filename, size_t label_col = 0);
    Dataset(const std::string filename, const std::vector<size_t>& ignore, size_t label_col = 0);

    Dataset split(float ratio);


    void one_hot_encode();   
    void normalize ();
    void standardize();


    void print_information();

};




class NeuralNetwork 
{   
private:

    bool imported = false;
    Loss loss_function_class;

    size_t lfunc_type;
    std::vector<size_t> afunc_type;


    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_fn lfunc;
    loss_derivative_fn lfunc_dx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_fn> afunc;
    std::vector<activation_fn> afunc_dx;

    std::vector<Matrix> weight_matrices;
    std::vector<Matrix> bias_matrices;


    void gradient_descent(const size_t steps, Dataset &ds, double lr, double lambda, size_t batch_size);
    void print_status(const size_t step, const size_t steps, const size_t batch_size, const size_t dataset_size, size_t& current_epoch);
    std::vector<Matrix> layer_outputs(const Matrix& input);
    

public:

    NeuralNetwork();

    void add_layer(const size_t neurons, activation_type  atype);
    void configure_loss_function(loss_type ltype);
    void set_loss_weights(const std::vector<float> w);
    void configure_input_layer(const size_t neurons);


    void initalise_random_weights(float begin = -0.1, float end = 0.1);
    void initalise_xavier_weights();
    void initalise_he_weights();

    void fit(const size_t epochs,Dataset &ds, optimizer_type ofunc, double lr, size_t batch_size = 64);
    void fit(const size_t epochs,Dataset &ds, optimizer_type ofunc, Hyperparameter &param);
    void fit(const size_t epochs,Dataset &ds, ADAM_Optimizer &adam);

    Matrix run(const Matrix& input);

    void performance(Dataset& ds, std::string name);
    void performance(Dataset& ds);
    
    void binary_confusion_matrix(Dataset& ds, const float threshold = 0.5);

    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename);
};








