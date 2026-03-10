#include "DeepModel.h"
#include <algorithm>
#include <fstream>    
#include <sstream>    
#include <string>    
#include <vector>     
#include <stdexcept>  
#include <cmath>

dataset<CPU>::dataset()
{
    
}

dataset<CPU>::dataset(const std::string filename, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        float current_output = -1; 
        size_t col = 0; 
        bool skip_row = false;

        while (std::getline(ss, cell, ',')) 
        { 
            try
            {
                float value = std::stof(cell);
                if (std::isnan(value)) {
                    skip_row = true;
                    break;
                }


                if (col == output_col) 
                    current_output = static_cast<float>(value); 
                else 
                    input_row.push_back(value); 
                ++col; 
            }
            catch(...)
            {
                skip_row = true;
                break;
            }

        }

        if(!input_row.empty() && !skip_row)
        {
            this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
            this->expected.push_back(matrix<CPU>(1,1, current_output)); 
        }
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
}

dataset<CPU>::dataset(const std::string filename, const std::vector<size_t>& ignore, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        float current_output = -1; 
        size_t col = 0; 
        bool skip_row = false;

        while (std::getline(ss, cell, ',')) 
        { 
            try
            {

                if (std::find(ignore.begin(), ignore.end(), col) != ignore.end()) 
                {   
                    ++col;
                    continue;
                }

                float value = std::stof(cell); 
                if (std::isnan(value)) {
                    skip_row = true;
                    break;
                }

                if (col == output_col) 
                    current_output = static_cast<float>(value); 
                else 
                    input_row.push_back(value); 
                ++col; 
            }
            catch(...)
            {
                skip_row = true;
                break;
            }

        }

        if(!input_row.empty() && !skip_row)
        {
            this->input.push_back(matrix<CPU>(input_row.size(), 1, input_row));
            this->expected.push_back(matrix<CPU>(1,1, current_output)); 
        }
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
    
}

dataset<CPU> dataset<CPU>::split(float ratio)
{

    if(0 > ratio || ratio > 1)
        throw std::runtime_error("dataset : split argument needs to be between 0 and 1 ");
        
    size_t split_point = this->input.size() * ratio;

    std::vector<matrix<CPU>> input_first_part(this->input.begin(), this->input.begin() + split_point);
    std::vector<matrix<CPU>> input_second_part(this->input.begin() + split_point, this->input.end());

    std::vector<matrix<CPU>> expected_first_part(this->expected.begin(), this->expected.begin() + split_point);
    std::vector<matrix<CPU>> expected_second_part(this->expected.begin() + split_point, this->expected.end());

    this->input = input_first_part;
    this->expected = expected_first_part;

    dataset ds;
    ds.input = input_second_part;
    ds.expected = expected_second_part;
    return ds;

}

void dataset<CPU>::one_hot_encode()
{
    if(this->expected[0].columns() != 1 || this->expected[0].rows() != 1)
        throw std::runtime_error("one_hot_encode : Wrong matrix output shape for one hot encoding. It needs to be 1x1xh."); 
    

    
    std::vector<matrix<CPU>> res;
    res.reserve(this->expected.size());

    std::vector<float> values;
    values.reserve(this->expected.size());
    

    for(matrix<CPU> &mat : this->expected)                               // ---
        values.push_back(mat[0]);
    
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    for(matrix<CPU> &mat : this->expected)
    {
        auto it = std::find(values.begin(), values.end(), mat[0]);
        int index = std::distance(values.begin(), it);

        matrix<CPU> _x = matrix<CPU>(values.size(), 1, 0);
        _x[index] = 1.0;
        res.push_back(_x);
    }

    this->expected = res;
    
}



void dataset<CPU>::normalize()
{
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for(matrix<CPU> vec : this->input )
    {
        float current_min = vec.min();
        float current_max = vec.max();         

        max = std::max(current_max, max);
        min = std::min(current_min, min);
    }

    if(max == min)
        throw std::runtime_error("normalize : All values of the dataset are the same, which results in a divison by zero.");

    for(matrix<CPU>& vec : this->input)
    {
        
        vec = vec - min;
        vec = vec * (1 / (max - min));
    }
    
    
}


void dataset<CPU>::standardize()
{
    
    size_t rows = this->input[0].rows();
    std::vector<float> means(0);
    means.reserve(rows);

    std::vector<float> sigma(0);
    sigma.reserve(rows);


    for(matrix<CPU> &vec : this->input)
        for(size_t r = 0; r < rows; r++ )
            means[r] += vec[r]; 
         
    for(size_t r = 0; r < rows; r++ )
        means[r]  = means[r] / this->input.size();



    for(matrix<CPU> &vec : this->input)
        for(size_t r = 0; r < rows; r++ )
            sigma[r] += (vec[r] - means[r]) *(vec[r] - means[r]); 
    
    for(size_t r = 0; r < rows; r++ )
        sigma[r]  = std::sqrt(sigma[r] / this->input.size());



    for(matrix<CPU> &vec : this->input)
    {
        for(size_t r = 0; r < rows; r++ )
            vec[r] = (vec[r] - means[r]) / sigma[r];   
    }
    
}

void dataset<CPU>::print_information()
{

    if(this->input.empty())
    {
        std::cout << "Dataset is empty." << std::endl; 
    }
    
    std::cout << "Samples : " << this->input.size() << std::endl;
    std::cout << "Input vector dim : " << this->input[0].rows() << std::endl;
    std::cout << "Output vector dim : " << this->expected[0].rows() << std::endl;

}
