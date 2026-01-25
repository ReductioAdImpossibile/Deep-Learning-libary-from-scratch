#pragma once
#include "model.h"
#include "matrix.h"
#include <string>

using activation_func = std::function<matrix<CPU>(matrix<CPU>&)>;
class model
{
protected:
    size_t input_layer_neurons;
    size_t output_layer_neurons;
    activation_func output_layer_afunc;

    std::vector<size_t> hidden_layer_neurons;
    std::vector<activation_func> hidden_layer_afuncs;

    model();

public:
    void configure_input_layer(const size_t neurons);
    void add_hidden_layer(const size_t neurons, activation_func  afunc);
    void confiugre_output_layer(const size_t neurons,  activation_func afunc);
};

template<>
class classificator<CPU> : public model
{   
private:
    std::vector<std::vector<float>> input;
    std::vector<std::vector<float>> expected;

public:
    classificator<CPU>();
    void load_csv(const std::string& filename, size_t label_col = 0);
};
