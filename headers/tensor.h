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
        
    std::vector<size_t> strides;
    alignas(ALIGN) std::vector<float> data;
    std::vector<size_t> shape;


    
public:

    tensor(const std::vector<size_t> &shape);
    tensor(const std::vector<size_t> &shape, float val);
    tensor(const std::vector<size_t> &shape, float begin, float end);

    const float& operator[](size_t index) const;
    float& operator[](size_t index);
    tensor operator%(const tensor &a) const;


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

    tensor sum(size_t axis);
    tensor slice(size_t axis);
    tensor reshape(const std::vector<size_t> &shape);

    static tensor batch_mat_mul(tensor &a, tensor &b);
    static bool equal_shape(const tensor &a, const tensor &b);

    static void hadamard(const tensor &a,const tensor &b, tensor &result);
    static void add(const tensor &a, const tensor &b, tensor &result);
    static void sub(const tensor &a, const tensor &b, tensor &result);
    static void scale(const tensor &a, const float value, tensor &result);
};

