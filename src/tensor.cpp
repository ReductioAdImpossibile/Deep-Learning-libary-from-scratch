
#include <iostream>
#include <vector>
#include <numeric>
#include "../headers/tensor.h"

#include <random>
#include <experimental/simd>
#include <omp.h>
#include <stdexcept>
#include <string>

std::string shape_to_string(const std::vector<size_t> &shape);

tensor::tensor(const std::vector<size_t> &_shape) : shape(_shape)
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

tensor::tensor(const std::vector<size_t> &_shape, float val) : tensor(_shape)
{
    #pragma omp parallel for
    for(int i{0}; i < this->data.size(); i++)
    {
        this->data[i] = val;
    }
}

tensor::tensor(const std::vector<size_t> &_shape, float begin, float end) : tensor(_shape)
{
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(begin, end);
    
    
    #pragma omp parallel for
    for(int i{0}; i < this->data.size(); i++)
    {
        this->data[i] = dist(gen);
    }
}



const float& tensor::operator[](size_t index) const
{
    return this->data[index];
}

float& tensor::operator[](size_t index)
{
    return this->data[index];
}

tensor tensor::operator%(const tensor &a) const
{
    tensor result(this->shape);
    tensor::hadamard(*this, a, result);
    return result;
}

tensor tensor::operator+(const tensor &a) const
{
    tensor result(this->shape);
    tensor::add(*this, a, result);
    return result;
}

tensor tensor::operator-(const tensor &a) const
{
    tensor result(this->shape);
    tensor::sub(*this, a, result);
    return result;
}

tensor tensor::operator*(const float &a) const
{
    tensor result(this->shape);
    tensor::scale(*this, a, result);
    return result;
}


float tensor::sum()
{
    //wrong!
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

std::vector<size_t> tensor::get_shape() const
{
    return this->shape;
}

tensor tensor::sum(size_t axis)
{
    // wrong!
    std::vector<size_t> res_shape = shape;
    res_shape.erase(res_shape.begin() + axis);
    
    tensor res(res_shape);
    std::vector<float>& res_vals = res.values(); 
    
    #pragma omp parallel for
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


float* tensor::raw()
{
    return this->data.data() ;
}

const float* tensor::raw() const
{
    return this->data.data() ;
}

float tensor::prod()
{

    // wrong!
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
    
}

float tensor::L2()
{

}

void tensor::print()
{
    for(auto val : data)
        std::cout << val << " ";
    std::cout << std::endl; 
}


size_t tensor::get_size() const
{
    return this->data.size();
}


std::vector<float>& tensor::values()
{
    return this->data;
}





// ------------------------- FAST OPERATIONS (static) --------------------------------
   
void tensor::hadamard(const tensor &a, const tensor &b, tensor &result)
{

    //wrong!
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
    {
        std::string msg = "Tensor shapes do not match for substraction. They need to be the same. Shapes:  \n ";
        msg += shape_to_string(a.get_shape()) + " (first parameter) \n" + shape_to_string(b.get_shape())  + " (second parameter) \n";
        msg += shape_to_string(result.get_shape()) + " (result) \n";
        throw std::runtime_error(msg);
    }

   
    int i{0};
    const int w = fsimd::size();
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();


    fsimd a_, b_, result_;

    #pragma omp parallel for
    for(; i + w < a.get_size(); i += w)
    {
        a_.copy_from(a_raw + i, std::experimental::element_aligned);
        b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
        result_ = a_ * b_;
        result_.copy_to(result_raw + i, std::experimental::element_aligned);
    }

    for(;i < a.get_size(); i++)
        result[i] = a[i] * b[i];

    
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



// -------------------------- FREE OPERATORS --------------------------------------
tensor operator*(float val, const tensor& a)
{
    return a * val;
}



// ------------------------ HELPING FUNCTIONS -------------------------------------

std::string shape_to_string(const std::vector<size_t> &shape)
{
    std::string res = "[ ";
    for(auto val : shape)
        res += std::to_string(val) + " ";
    
    res += "]";
    return res;
}