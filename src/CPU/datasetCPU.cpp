#include "DeepModel.h"
#include "modelCPU.h"



dataset<CPU>::dataset()
{
    
}

dataset<CPU>::dataset(const std::string filename, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;

        float current_output = -1; 
        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell); 
            if (col == output_col) 
                current_output = static_cast<float>(value); 
            else 
                input_row.push_back(value); 
            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(1,1, current_output)); 
    }
    
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> output_cols)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        std::vector<float> output_row;

        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell); 
            if (std::find(output_cols.begin(), output_cols.end(), col) != output_cols.end()) 
                output_row.push_back(static_cast<float>(value)); 
            else 
                input_row.push_back(value); 
            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(output_row.size(), 1, output_row));
    }
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t> input_cols, const std::vector<size_t> output_cols)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        std::vector<float> output_row;

        size_t col = 0; 
        while (std::getline(ss, cell, ',')) 
        { 
            float value = std::stof(cell);

            if (std::find(output_cols.begin(), output_cols.end(), col) != output_cols.end()) 
                output_row.push_back(static_cast<float>(value)); 
            else if(std::find(input_cols.begin(), input_cols.end(), col) != input_cols.end()) 
                input_row.push_back(value); 

            ++col; 
        }

        this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
        this->expected.push_back(matrix<CPU>(output_row.size(), 1, output_row));
    }
}


void dataset<CPU>::one_hot_encode()
{
    
}

void dataset<CPU>::normalize()
{
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for(matrix<CPU> vec : this->input )
    {
        float current_min = *std::min_element(vec.begin(), vec.end());
        float current_max = *std::max_element(vec.begin(), vec.end());

        max = std::max(current_max, max);
        min = std::min(current_min, min);
    }

    if(max == min)
        throw std::runtime_error("max values equal min values, which results in a divison by zero");

    for(matrix<CPU>& vec : this->input)
    {
        
        vec = vec - min;
        vec = vec * (1 / (max - min));
    }
    
    
}

void dataset<CPU>::standardize()
{
}
