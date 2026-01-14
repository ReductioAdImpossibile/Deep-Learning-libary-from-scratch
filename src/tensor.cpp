
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

float& tensor::operator[](uint64_t index)
{
    return this->data[index];
}



tensor::tensor(const std::vector<uint64_t> &_shape) : shape(_shape)
{
    this->strides.resize(shape.size());
    size_t prod = 1;
    for(int i = strides.size()-1; i >= 0; i--)
    {
        this->strides[i] = prod;
        prod *= this->shape[i];
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
        this->data[i] = val;
    }
}

tensor::tensor(const std::vector<uint64_t> &_shape, float begin, float end) : tensor(_shape)
{
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(begin, end);
    
    
    #pragma omp parallel for
    for(int i{0}; i < this->data.size(); i++)
    {
        this->data[i] = dist(gen);
    }
}



float tensor::sum()
{
    float* raw = this->raw();
    fsimd acc1_(0.0f), acc2_(0.0f);
    fsimd val1_, val2_;

    int i{0};
    const int w = fsimd::size();

    #pragma omp parallel for reduction(+:acc1_, acc2_)    
    for(;  i + 2 * w <= this->data.size(); i += 2 * w ) 
    {
        val1_.copy_from(raw + i, std::experimental::element_aligned);
        val2_.copy_from(raw + i + w, std::experimental::element_aligned);

        acc1_ += val1_;
        acc2_ += val2_;
    }
    
    float res = 0;
    for(int j{0}; j < w; j++)
        res += acc1_[j] + acc2_[j];

    for(; i < this->data.size(); i++)
        res += this->data[i];

    return res;
}

std::vector<uint64_t> tensor::get_shape()
{
    return this->shape;
}

tensor tensor::sum(size_t axis)
{
    std::vector<uint64_t> res_shape = shape;
    res_shape.erase(res_shape.begin() + axis);
    
    tensor res(res_shape);
    std::vector<float>& res_vals = res.values(); 
    
    #pragma omp parallel
    for(size_t i{0}; i < res_vals.size(); i++ )
    {

        size_t tmp = i;
        size_t base = 0;
        for(size_t j{0}; j < res_shape.size(); j++)
        {
            size_t idx = tmp % this->shape[j];
            tmp /= this->shape[j];

            size_t op = (j < axis) ? j : j + 1; 
            base += idx * this->strides[op];
        }

        float sum = 0; 
        for(size_t k{0}; k < this->shape[axis]; k++)
        {
            size_t idx = base + this->strides[axis] * k;
            sum += this->data[idx];
        }

        res[i] = sum;
    }

    return res;
}

float tensor::prod()
{
    float* raw = this->raw();
    fsimd acc1_(1.0f), acc2_(1.0f);
    fsimd val1_, val2_;


    int i{0};
    const int w = fsimd::size();

    #pragma omp parallel for reduction(*:acc1_, acc2_)    
    for(;  i + 2 * w <= this->data.size(); i += 2 * w ) 
    {
        val1_.copy_from(raw + i, std::experimental::element_aligned);
        val2_.copy_from(raw + i + w, std::experimental::element_aligned);

        acc1_ *= val1_;
        acc2_ *= val2_;
    }
    
    float res = 1;
    for(int j{0}; j < w; j++)
    {   
        res *= (acc1_[j] * acc2_[j]);
    }

    for(; i < this->data.size(); i++)
        res *= this->data[i];

    return res;

}

float tensor::max()
{
    float max_val = -std::numeric_limits<float>::infinity();
    size_t n = this->data.size();

    #pragma omp parallel for reduction(max : max_val)
    for(size_t i = 0; i < n; i++)
    {
        if(this->data[i] > max_val)
            max_val = this->data[i];
    }

    return max_val;
}

float tensor::min()
{
    float min_val = -std::numeric_limits<float>::infinity();
    size_t n = this->data.size();

    #pragma omp parallel for reduction(min : min_val)
    for(size_t i = 0; i < n; i++)
    {
        if(this->data[i] < min_val)
            min_val = this->data[i];
    }

    return min_val;    
}

float tensor::avg()
{
    return this->sum() / this->data.size();
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

std::vector<float>& tensor::values()
{
    return this->data;
}
