
#include <iostream>
#include <vector>
#include <numeric>
#include "../headers/tensorCPU.h"

#include <random>
#include <experimental/simd>
#include <omp.h>
#include <stdexcept>
#include <string>

std::string shape_to_string(const std::vector<size_t> &shape);

tensor<CPU>::tensor(const std::vector<size_t> &_shape) : shape(_shape)
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

tensor<CPU>::tensor(const std::vector<size_t> &_shape, float val) : tensor<CPU>(_shape)
{
    #pragma omp parallel for
    for(int i = 0; i < this->data.size(); i++)
    {
        this->data[i] = val;
    }
}

tensor<CPU>::tensor(const std::vector<size_t> &_shape, float begin, float end) : tensor<CPU>(_shape)
{
    
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(begin, end);
    
    
    #pragma omp parallel for
    for(int i = 0; i < this->data.size(); i++)
    {
        this->data[i] = dist(gen);
    }
}



const float& tensor<CPU>::operator[](size_t index) const
{
    return this->data[index];
}

float& tensor<CPU>::operator[](size_t index)
{
    return this->data[index];
}

tensor<CPU> tensor<CPU>::operator%(const tensor<CPU> &a) const
{
    tensor result(this->shape);
    tensor::hadamard(*this, a, result);
    return result;
}

tensor<CPU> tensor<CPU>::operator+(const tensor<CPU> &a) const
{
    tensor result(this->shape);
    tensor::add(*this, a, result);
    return result;
}

tensor<CPU> tensor<CPU>::operator-(const tensor<CPU> &a) const
{
    tensor result(this->shape);
    tensor::sub(*this, a, result);
    return result;
}

tensor<CPU> tensor<CPU>::operator*(const float &a) const
{
    tensor result(this->shape);
    tensor::scale(*this, a, result);
    return result;
}


float tensor<CPU>::sum()
{
    float* raw = this->raw();
    const size_t n = this->data.size();
    const size_t m = omp_get_max_threads();
    const size_t w = fsimd::size();

    float global_sum{0.0f};
    #pragma omp parallel                   
    {
        fsimd simd_sum(0.0f);
        float local_sum{0.0f};

        int num = omp_get_thread_num();
        size_t start =  num * (n / m); 
        size_t end = num == m-1 ? n : (num + 1) * (n/m);

        for (size_t i{start}; i + w <= end; i += w)
        {
            fsimd v;
            v.copy_from(raw + i, std::experimental::element_aligned);
            simd_sum += v;
        }

        for (size_t  j{0}; j < w; j++)
            local_sum += simd_sum[j];

        #pragma omp atomic
        global_sum += local_sum;
    }

    for (size_t i = (n / w) * w; i < n; i++)        // i = n - (n mod w)
        global_sum += raw[i];

    return global_sum;
}

std::vector<size_t> tensor<CPU>::get_shape() const
{
    return this->shape;
}

tensor<CPU> tensor<CPU>::sum(size_t axis)
{
    // wrong!
    std::vector<size_t> res_shape = shape;
    res_shape.erase(res_shape.begin() + axis);
    
    tensor res(res_shape);
    std::vector<float>& res_vals = res.values(); 
    
    #pragma omp parallel for
    for(size_t i = 0; i < res_vals.size(); i++ )
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


float* tensor<CPU>::raw()
{
    return this->data.data() ;
}

const float* tensor<CPU>::raw() const
{
    return this->data.data() ;
}

float tensor<CPU>::prod()
{

}

float tensor<CPU>::max()
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

float tensor<CPU>::min()
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

float tensor<CPU>::avg()
{
    return this->sum() / this->data.size();
}

float tensor<CPU>::L1()
{
    
}

float tensor<CPU>::L2()
{

}

void tensor<CPU>::print()
{
    for(auto val : data)
        std::cout << val << " ";
    std::cout << std::endl; 
}


size_t tensor<CPU>::get_size() const
{
    return this->data.size();
}


std::vector<float>& tensor<CPU>::values()
{
    return this->data;
}





// ------------------------- FAST OPERATIONS (static) --------------------------------
   
void tensor<CPU>::hadamard(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
    {
        std::string msg = "Tensor shapes do not match for substraction. They need to be the same. Shapes:  \n ";
        msg += shape_to_string(a.get_shape()) + " (first parameter) \n" + shape_to_string(b.get_shape())  + " (second parameter) \n";
        msg += shape_to_string(result.get_shape()) + " (result) \n";
        throw std::runtime_error(msg);
    }

   
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();

    const size_t n = a.get_size();
    const size_t w = fsimd::size();

    #pragma omp parallel for
    for(int i = 0; i <  n - w; i += w)
    {
        fsimd a_, b_, result_;
        a_.copy_from(a_raw + i, std::experimental::element_aligned);
        b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
        result_ = a_ * b_;
        result_.copy_to(result_raw + i, std::experimental::element_aligned);
    }


    for(int j = (n/w) * w ;j < n; j++)
        result_raw[j] = a_raw[j] * b_raw[j];
    
    
    
}

void tensor<CPU>::add(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result)
{
}

void tensor<CPU>::sub(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result)
{

}

void tensor<CPU>::scale(const tensor<CPU> &a, const float value, tensor<CPU> &result)
{

}



// -------------------------- FREE OPERATORS --------------------------------------
tensor<CPU> operator*(float val, const tensor<CPU> &a)
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