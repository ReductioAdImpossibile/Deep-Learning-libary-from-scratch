#include "modelCPU.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

model::model()
{
    this->input_layer_neurons = 0;
    this->output_layer_neurons = 0;
}

void model::configure_input_layer(const size_t neurons) 
{   
    this->input_layer_neurons = neurons;
}

void model::add_hidden_layer(const size_t neurons, activation_func afunc) 
{
    hidden_layer_neurons.push_back(neurons);
    hidden_layer_afuncs.push_back(afunc);
}

void model::confiugre_output_layer(const size_t neurons, activation_func afunc)
{
    this->output_layer_neurons = neurons;
    this->output_layer_afunc = afunc;
}



classificator<CPU>::classificator() : model()
{}

void classificator<CPU>::load_csv(const std::string &filename, size_t label_col ) 
{
    if (output_layer_neurons == 0)
        throw std::runtime_error("Output layer not configured. Run configure_output_layer(neurons, activation func) first.");

    if (input_layer_neurons == 0)
        throw std::runtime_error("Input layer not configured. Run configure_input_layer(neurons) first.");

    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open CSV file: " + filename);

    input.clear();
    expected.clear();

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        std::vector<float> input_row;
        float current_label = -1;

        size_t col = 0;
        while (std::getline(ss, cell, ','))
        {
            float value = std::stof(cell);

            if (col == label_col)
                current_label = value;
            else
                input_row.push_back(value);
            ++col;
        }
        
        if (input_row.size() != input_layer_neurons)
        {
            throw std::runtime_error(
                "Input size mismatch in CSV row. Expected " +
                std::to_string(input_layer_neurons) +
                ", got " + std::to_string(input_row.size()));
        }

        input.push_back(input_row);

      
        if (current_label < 0 || current_label >= static_cast<int>(output_layer_neurons))
            throw std::runtime_error("Label out of range: " + std::to_string(current_label));

        std::vector<float> one_hot(output_layer_neurons, 0.0f);
        one_hot[static_cast<int>(current_label)] = 1.0f;
        expected.push_back(one_hot);
      
    }

    if (input_layer_neurons == 0 && !input.empty())
        input_layer_neurons = input.front().size();
}
