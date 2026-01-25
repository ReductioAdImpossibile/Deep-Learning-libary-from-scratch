
#include <iostream>
#include <vector>
#include <numeric>
#include "tensorCPU.h"

#include <random>
#include <experimental/simd>
#include <omp.h>
#include <stdexcept>
#include <string>

std::string shape_to_string(const std::vector<size_t> &shape);

tensor<CPU>::tensor()
{}

tensor<CPU>::tensor(const tensor<CPU> &other) 
    : strides(other.strides), shape(other.shape), n(other.n)
{
    this->data = (float*) _mm_malloc(n * sizeof(float), ALIGN);
    std::copy(other.data, other.data + n, data);
}

tensor<CPU>::tensor(const std::vector<size_t> &_shape) : shape(_shape)
{

    this->strides.resize(shape.size());
    size_t prod = 1;
    for(int i = strides.size()-1; i >= 0; i--)
    {
        this->strides[i] = prod;
        prod *= this->shape[i];
    }

    this->n  =  std::accumulate(shape.begin(), shape.end(), 1,  std::multiplies<int>());
    this->data = (float*) _mm_malloc(n * sizeof(float), ALIGN);

    if (!data) throw std::bad_alloc();

    
}

tensor<CPU>::tensor(const std::vector<size_t> &_shape, float val) : tensor<CPU>(_shape)
{

    const size_t limit = (n/w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v(val);
        v.copy_to(this->data + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < n; i++)
        this->data[i] = val;

}

tensor<CPU>::tensor(const std::vector<size_t> &_shape, float begin, float end) : tensor<CPU>(_shape)
{
    #pragma omp parallel
    {
        std::mt19937 gen(
            std::random_device{}() + omp_get_thread_num()
        );
        std::uniform_real_distribution<float> dist(begin, end);

        #pragma omp for
        for (size_t i = 0; i < this->n; ++i)
        {
            this->data[i] = dist(gen);
        }
    }

}

tensor<CPU>::~tensor() {
    if(data) _mm_free(data);
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
    float global_sum = 0.0f;

    #pragma omp parallel reduction(+:global_sum)
    {
        fsimd simd_sum(0.0f);
        double local_sum = 0.0;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_max_threads();
        
        const size_t start = tid * (this->n / nthreads);
        const size_t end   = tid == nthreads-1 ? n : (tid+1)*(this->n/nthreads);
        const size_t limit = start + ((end - start) / w) * w;

        for (size_t i = start; i < limit; i += w)
        {
            fsimd v;
            v.copy_from(this->data + i, std::experimental::element_aligned);
            simd_sum += v;
        }

        local_sum = stdx::reduce(simd_sum, std::plus{});

        
        for(size_t j = limit; j < end; j++)
            local_sum += this->data[j];

        global_sum += static_cast<float>(local_sum);
    }

    return global_sum;
}

std::vector<size_t> tensor<CPU>::get_shape() const
{
    return this->shape;
}

tensor<CPU> tensor<CPU>::sum(size_t axis)
{
    std::vector<size_t> res_shape = shape;
    res_shape.erase(res_shape.begin() + axis);
    
    tensor res(res_shape);
    std::vector<float> res_vals = res.vector(); 
    
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
    return this->data;
}

float* tensor<CPU>::raw() const
{
    return this->data;
}

float tensor<CPU>::index(const std::vector<int> &indices)
{
    size_t p = 0;
    for(int i = 0; i < strides.size(); i++)
        p += indices[i] * strides[i]; 
    return data[p];
}

float tensor<CPU>::prod()
{
    float global_log_sum = 0.0f;

    #pragma omp parallel reduction(+:global_log_sum)
    {
        fsimd simd_log_sum(0.0f);
        double local_log_sum = 0.0;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_max_threads();
        const size_t start = tid * (this->n / nthreads);
        const size_t end   = tid == nthreads-1 ? n : (tid+1)*(this->n/nthreads);
        const size_t limit = start + ((end - start) / w) * w;

        for (size_t i = start; i < limit; i += w)
        {
            fsimd v;
            v.copy_from(this->data + i, std::experimental::element_aligned);
            
            v = stdx::log(v);
            simd_log_sum += v;
        }

        local_log_sum = stdx::reduce(simd_log_sum, std::plus{});

        
        for(size_t j = limit; j < end; j++)
            local_log_sum += std::log(this->data[j]);


        global_log_sum += static_cast<float>(local_log_sum);
    }
    return std::exp(global_log_sum);
}

float tensor<CPU>::max()
{
    const float min_float = -std::numeric_limits<float>::infinity();

    fsimd max_simd(min_float);
    fsimd v;

    size_t i{0};
    const size_t limit = (n/w) * w;
    float max = min_float;

    for(; i < limit ; i += w)
    {
        v.copy_from(this->data + i, std::experimental::element_aligned);
        where(v > max_simd, max_simd) = v; 
    }
    
    for(int j{0}; j < w; j++)
    {
        if(max_simd[j] > max)
            max = max_simd[j];
    }

    for(; i < this->n; i++)
    {
        if(this->data[i] > max)
            max = this->data[i];
    }

    return max;
}

float tensor<CPU>::min()
{
    const float max_float = std::numeric_limits<float>::infinity();

    fsimd min_simd(max_float);
    fsimd v;
    size_t i{0};
    const size_t limit = (n/w) *w;
    float min = max_float;

    for(; i < limit; i += w)
    {
        v.copy_from(this->data + i, std::experimental::element_aligned);
        where(v < min_simd, min_simd) = v; 
    }
    
    for(int j{0}; j < w; j++)
    {
        if(min_simd[j] < min)
            min = min_simd[j];
    }

    for(; i < this->n; i++)
    {
        if(this->data[i] < min)
            min = this->data[i];
    }

    return min; 
}

float tensor<CPU>::avg()
{
    return this->sum() / this->n;
}

float tensor<CPU>::L1()
{
    float global_sum = 0.0f;

    #pragma omp parallel reduction(+:global_sum)
    {
        fsimd simd_sum(0.0f);
        double local_sum = 0.0;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_max_threads();
        size_t start = tid * (this->n / nthreads);
        size_t end   = tid == nthreads-1 ? this->n : (tid+1)*(this->n/nthreads);
        size_t limit = start + ((end - start) / w) * w;

        for (size_t i = start; i < limit; i += w)
        {
            fsimd v;
            v.copy_from(this->data + i, std::experimental::element_aligned);
            where(v < 0, v) = -v;
            simd_sum += v;
        }

        local_sum = stdx::reduce(simd_sum, std::plus{});
           
        for(size_t j = limit; j < end; j++)
            local_sum += std::abs(this->data[j]);
        
        global_sum += static_cast<float>(local_sum);
    }

    return global_sum;
}

float tensor<CPU>::L2()
{
    float global_sum = 0.0f;

    #pragma omp parallel reduction(+:global_sum)
    {
        fsimd simd_sum(0.0f);
        double local_sum = 0.0;

        int tid = omp_get_thread_num();
        int nthreads = omp_get_max_threads();
        size_t start = tid * (this->n / nthreads);
        size_t end   = tid == nthreads-1 ? n : (tid+1)*(this->n/nthreads);
        size_t limit = start + ((end - start) / w) * w;

        for (size_t i = start; i  < limit; i += w)
        {
            fsimd v;
            v.copy_from(this->data + i, std::experimental::element_aligned);
            simd_sum += v * v;
        }
        
        local_sum = stdx::reduce(simd_sum, std::plus{});

        for(size_t j = limit; j < end; j++)
        {
            local_sum += this->data[j] * this->data[j];
        }
        global_sum += static_cast<float>(local_sum);
    }

    return std::sqrt(global_sum);
}

void tensor<CPU>::print()
{
    for(int i = 0; i < this->n; i++)
        std::cout << this->data[i] << " ";
    std::cout << std::endl; 
}

void tensor<CPU>::set(float val)
{
    const ssize_t limit = (n / w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v(val);
        v.copy_to(data + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < this->n; i++)
        data[i] = val;
}


void tensor<CPU>::set_zero()
{
    this->set(0);
}

size_t tensor<CPU>::size() const
{
    return this->n;
}


std::vector<float> tensor<CPU>::vector() 
{   
    return std::vector<float>(this->data, this->data + this->n);
}

bool tensor<CPU>::equal_shape(const tensor<CPU> &a, const tensor<CPU> &b)
{
    return (a.get_shape() == b.get_shape());
}

// ------------------------- FAST OPERATIONS (static) --------------------------------

void tensor<CPU>::hadamard(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result) 
{
    
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the hadamard product. They need to be the same");
    
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(size_t i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ * b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = limit ;j < n; j++)
        result_raw[j] = a_raw[j] * b_raw[j];

}

void tensor<CPU>::add(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the tensor addition. They need to be the same");
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit ; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ + b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = (n/w) * w ;j < n; j++)
        result_raw[j] = a_raw[j] + b_raw[j];

 
}

void tensor<CPU>::sub(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the tensor addition. They need to be the same");
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;
    

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ - b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = limit ;j < n; j++)
        result_raw[j] = a_raw[j] * b_raw[j];

}

void tensor<CPU>::scale(const tensor<CPU> &a, const float value, tensor<CPU> &result)
{
    if( !(a.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the hadamard product. They need to be the same");
    
    
    const float* a_raw = a.raw();  
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);    
            result_ = a_ * value;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = (n/w) * w ;j < n; j++)
        result_raw[j] = a_raw[j] * value;

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