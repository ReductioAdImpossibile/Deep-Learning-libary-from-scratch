
#include <iostream>
#include <vector>
#include <numeric>
#include "../headers/tensor.h"

#include <random>
#include <experimental/simd>
#include <omp.h>


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


tensor::tensor(const std::vector<uint64_t> &_shape) : shape(_shape)
{
    this->strides.resize(shape.size());
    size_t prod = 1;
    for(int i = strides.size()-1; i >= 0; i--)
    {
        this->strides.at(i) = prod;
        prod *= this->shape.at(i);
    }

    data.resize(
        std::accumulate(shape.begin(), shape.end(), 1,  std::multiplies<int>())
    );

    omp_set_num_threads(10); 
}

tensor::tensor(const std::vector<uint64_t> &_shape, float val) : tensor(_shape)
{
    #pragma omp parallel for
    for(int i{0}; i < this->data.size(); i++)
    {
        this->data.at(i) = val;
    }
}

tensor::tensor(const std::vector<uint64_t> &_shape, float begin, float end) : tensor(_shape)
{
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(begin, end);
    
    
    #pragma omp parallel for
    for(int i{0}; i < this->data.size(); i++)
    {
        this->data.at(i) = dist(gen);
    }
}


float tensor::sum()
{

    float* raw = this->raw();
    fsimd sum1_(0.0f), sum2_(0.0f);
    fsimd val1_, val2_;

    int i{0};
    const int w = fsimd::size();


    
    //#pragma omp parallel for reduction(+:sum1_, sum2_)    
    for(; 2 * i < this->data.size(); i += 2 * w ) 
    {
        val1_.copy_from(raw + i, std::experimental::element_aligned);
        val2_.copy_from(raw + i + w, std::experimental::element_aligned);

        sum1_ += val1_;
        sum2_ += val2_;
    }
    

    float res = 0;
    for(int j{0}; j < w; j++)
        res += sum1_[j] + sum2_[j];

    for(; i < this->data.size(); i++)
        res += this->data.at(i);

    return res;
}

std::vector<uint64_t> tensor::get_shape()
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






