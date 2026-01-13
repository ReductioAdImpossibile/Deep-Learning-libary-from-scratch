#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <experimental/simd>



namespace stdx = std::experimental;
using fsimd = stdx::native_simd<float>;
constexpr std::size_t ALIGN = alignof(fsimd);

class tensor
{
private:
        
    std::vector<unsigned int> strides;
    alignas(ALIGN) std::vector<float> data;
    std::vector<uint64_t> shape;

    static void hadamard(const tensor &a, const tensor &b, tensor &result);
    static void add(const tensor &a, const tensor &b, tensor &result);
    static void sub(const tensor &a, const tensor &b, tensor &result);
    static void scale(const tensor &a, const float value, tensor &result);

public:

    
    
    tensor(const std::vector<uint64_t> &shape);
    
    tensor(const std::vector<uint64_t> &shape, float val);

    tensor(const std::vector<uint64_t> &shape, float begin, float end);


    float* raw();
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

    std::vector<uint64_t> get_shape();

    tensor sum(size_t axis);
    tensor slice(size_t axis);
    tensor reshape(const std::vector<uint64_t> &shape);

    static tensor batch_mat_mul(const tensor &a, const tensor &b);
};