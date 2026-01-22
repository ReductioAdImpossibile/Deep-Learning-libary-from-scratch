#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <experimental/simd>
#include "backend.h"
#include "tensor.h"


namespace stdx = std::experimental;
using fsimd = stdx::native_simd<float>;
using fmask = fsimd::mask_type;

constexpr std::size_t ALIGN = alignof(fsimd);

template<>
class tensor<CPU>
{
private:
        
    std::vector<size_t> strides;
    alignas(ALIGN) std::vector<float> data;
    std::vector<size_t> shape;

public:

    tensor(const std::vector<size_t> &shape);
    tensor(const std::vector<size_t> &shape, float val);
    tensor(const std::vector<size_t> &shape, float begin, float end);

    const float& operator[](size_t index) const;
    float& operator[](size_t index);
    
    tensor<CPU> operator%(const tensor<CPU> &a) const;
    tensor<CPU> operator+(const tensor<CPU> &a) const;
    tensor<CPU> operator-(const tensor<CPU> &a) const;
    tensor<CPU> operator*(const float &a) const;

    float* raw();
    const float* raw() const; 
    
    float index(const std::vector<int> &indices); 

    float sum();
    float prod();
    float max();
    float min();
    float avg();
    float L1();
    float L2();

    
    void print();
    void set(float val);
    void set_zero();

    size_t get_size() const;
   

    std::vector<size_t> get_shape() const;
    std::vector<float>& values();   

    tensor<CPU> sum(size_t axis);
    tensor<CPU> slice(size_t axis);
    tensor<CPU> reshape(const std::vector<size_t> &shape);

    //static tensor<CPU> batch_mat_mul(tensor<CPU> &a, tensor<CPU> &b);
    static bool equal_shape(const tensor<CPU> &a, const tensor<CPU> &b);

    static void hadamard(const tensor<CPU> &a,const tensor<CPU> &b, tensor<CPU> &result);
    static void add(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result);
    static void sub(const tensor<CPU> &a, const tensor<CPU> &b, tensor<CPU> &result);
    static void scale(const tensor<CPU> &a, const float value, tensor<CPU> &result);
};

tensor<CPU> operator*(float val, const tensor<CPU>& a);