#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <experimental/simd>

namespace stdx = std::experimental;
using fsimd = stdx::native_simd<float>;
using fmask = fsimd::mask_type;

constexpr std::size_t ALIGN = alignof(fsimd);
constexpr ssize_t w = fsimd::size();

template<>
class matrix<CPU> 
{
protected:
    size_t c, r;
    float* data;
    size_t n;
    std::vector<size_t> shape;

public:
    
    matrix<CPU>();
    matrix<CPU>(const matrix<CPU>& other);
    matrix<CPU>(const std::vector<size_t> &shape);
    matrix<CPU>(const std::vector<size_t> &shape, float val);
    matrix<CPU>(const std::vector<size_t> &shape, float start, float end);
    ~matrix<CPU>();

    const float& operator[](size_t index) const;
    float& operator[](size_t index);
    matrix<CPU> operator%(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const float &a) const;
    matrix<CPU> operator-(const matrix<CPU> &a) const;
    matrix<CPU> operator-(const float &a) const;
    matrix<CPU> operator*(const float &a) const;
    matrix<CPU> operator*(const matrix<CPU> &a) const;

    float* raw();
    float* raw() const; 

    static void mat_mul(const matrix& a, const matrix &b, matrix& result);
    static void mat_mul_transposed(const matrix& a, const matrix &b, matrix& result);

    void transpose();
    static void transpose(const matrix &a, matrix &result);

    std::vector<size_t> get_shape() const;
    size_t rows() const;
    size_t columns() const;
    size_t size() const;

    void print();
    void set(float val);

    static bool equal_shape(const matrix<CPU> &a, const matrix<CPU>  &b);
    static void hadamard(const matrix<CPU>  &a,const matrix<CPU>  &b, matrix<CPU>  &result);
    static void add(const matrix<CPU>  &a, const matrix<CPU>  &b, matrix<CPU>  &result);
    static void sub(const matrix<CPU>  &a, const matrix<CPU>  &b, matrix<CPU>  &result);
    static void scale(const matrix<CPU>  &a, const float value, matrix<CPU>  &result);    
};

matrix<CPU> operator*(float val, const matrix<CPU>& a);
matrix<CPU> operator+(float val, const matrix<CPU>& a);
matrix<CPU> operator-(float val, const matrix<CPU>& a);
