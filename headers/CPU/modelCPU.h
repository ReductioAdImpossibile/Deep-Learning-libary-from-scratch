#pragma once
#include "DeepModel.h"
#include "activationCPU.h"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>





template<>
class neuralnetwork<CPU> 
{   
private:

    bool imported = false;
    loss<CPU> loss_function_class;

    size_t lfunc_type;
    std::vector<size_t> afunc_type;


    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_fn lfunc;
    loss_derivative_fn lfunc_dx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_fn> afunc;
    std::vector<activation_fn> afunc_dx;

    std::vector<matrix<CPU>> weight_matrices;
    std::vector<matrix<CPU>> bias_matrices;


    void gradient_descent(const size_t steps, dataset &ds, double lr, double lambda, size_t batch_size);
    
    std::vector<matrix<CPU>> layer_outputs(const matrix<CPU>& input);
    matrix<CPU> run(const matrix<CPU>& input);

    

public:

    neuralnetwork<CPU>();

    void add_layer(const size_t neurons, activation_type  atype);
    void configure_loss_function(loss_type ltype);
    void set_loss_weights(const std::vector<float> w);
    void configure_input_layer(const size_t neurons);


    void initalise_random_weights(float begin = -0.1, float end = 0.1);
    void initalise_xavier_weights();
    void initalise_he_weights();

    void fit(const size_t epochs,dataset &ds, optimizer_type ofunc, double lr, size_t batch_size = 64);
    void fit(const size_t epochs,dataset &ds, optimizer_type ofunc, hyperparameter<CPU> &param);
    void fit(const size_t epochs,dataset &ds, adam_optimizer<CPU> &adam);



    void performance(dataset& ds, std::string name);
    void performance(dataset& ds);
    
    void binary_confusion_matrix(dataset& ds, const float threshold = 0.5);

    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename);
};
