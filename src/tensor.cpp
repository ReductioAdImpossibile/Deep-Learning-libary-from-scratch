
#include <iostream>
#include <vector>
#include <numeric>
#include "../headers/tensor.h"
#include <random>
#include <experimental/simd>



void tensor::hadamard(const tensor &a, const tensor &b, tensor &result)
{

}

void tensor::add(const tensor &a, const tensor &b, tensor &result)
{

}

void tensor::sub(const tensor &a, const tensor &b, tensor &result)
{

}

void tensor::scale(const tensor &a, const float value, tensor &result)
{

}


tensor::tensor(const std::vector<int> &_shape) : shape(_shape)
{
    this->strides.resize(shape.size());
    size_t prod = 1;
    for(int i = strides.size()-1; i >= 0; i--)
    {
        this->strides.at(i) = prod;
        prod *= this->shape.at(i);
    }

    data.resize(
        std::accumulate(shape.begin(), shape.end(), 1)
    );
}

tensor::tensor(const std::vector<int> &_shape, float val) : tensor(_shape)
{
    for(int i{0}; i < this->data.size(); i++)
        this->data.at(i) = val;
}

tensor::tensor(const std::vector<int> &_shape, float begin, float end) : tensor(_shape)
{
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(begin, end);
    
    for(int i{0}; i < this->data.size(); i++)
        this->data.at(i) = dist(gen);
}


float tensor::sum()
{

    //float_simd sum_simd = float_simd::
    for(int i{0}; i < this->data.size();++i)
    {

    }

    return 0.0f;
}

std::vector<int> tensor::get_shape()
{
    return this->shape;
}

tensor tensor::sum(size_t axis)
{
    return tensor({});
}

float tensor::prod()
{
    return 0.0f;
}

float tensor::max()
{
    return 0.0f;
}

float tensor::min()
{
    return 0.0f;
}

float tensor::avg()
{
    return 0.0f;
}

float tensor::L1()
{
    return 0.0f;
}

float tensor::L2()
{
    return 0.0f;
}

void tensor::print()
{
    for(auto val : data)
        std::cout << val << " ";
    std::cout << std::endl; 
}

float* tensor::raw()
{
    return this->data.data() ;
}






