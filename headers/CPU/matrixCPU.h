#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>

template<>
class matrix<CPU> 
{
protected:
    size_t c, r;
    std::vector<float> data;
    size_t n;

public:
    
    matrix<CPU>();
    matrix<CPU>(const matrix<CPU>& other);
    matrix<CPU>(matrix<CPU>&& other) noexcept;
    matrix<CPU>(const size_t rows, const size_t columns);
    matrix<CPU>(const size_t rows, const size_t columns, const std::vector<float>& values);
    matrix<CPU>(const size_t rows, const size_t columns, float val);
    matrix<CPU>(const size_t rows, const size_t columns, float start, float end);

    matrix<CPU>& operator=(const matrix<CPU>& other);
    matrix<CPU>& operator=(matrix<CPU>&& other) noexcept;

    const float& operator[](size_t index) const;
    float& operator[](size_t index);
    matrix<CPU> operator%(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const float &a) const;
    matrix<CPU> operator-(const matrix<CPU> &a) const;
    matrix<CPU> operator-(const float &a) const;
    matrix<CPU> operator*(const float &a) const;
    matrix<CPU> operator*(const matrix<CPU> &a) const;
    matrix<CPU> operator+=(const matrix<CPU> &a);
    matrix<CPU> operator-=(const matrix<CPU> &a);


    static void mat_mul(const matrix& a, const matrix &b, matrix& result);                 
    static void mat_mul_transposed(const matrix& a, const matrix &b, matrix& result);       


    matrix<CPU> transpose();                                                                
    static void transpose(const matrix &a, matrix &result);

    size_t rows() const;
    size_t columns() const;
    size_t size() const;
    std::vector<float>& raw();
    std::vector<float> raw_copy();

    double L1();
    double L2();
    size_t argmax();
    size_t argmin();

    void print();
    void print_size();
    void set(float val);
    void insert_row(size_t row_pos, float val );
    void remove_row(size_t row_pos);

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }


    static void hadamard(const matrix<CPU>  &a,const matrix<CPU>  &b, matrix<CPU>  &result);
    static void add(const matrix<CPU>  &a, const matrix<CPU>  &b, matrix<CPU>  &result);
    static void sub(const matrix<CPU>  &a, const matrix<CPU>  &b, matrix<CPU>  &result);
    static void scale(const matrix<CPU>  &a, const float value, matrix<CPU>  &result);
    
    static matrix<CPU> sqrt(const matrix<CPU> &a);
    static matrix<CPU> square(const matrix<CPU> &a);
    static matrix<CPU> reciprocal(const matrix<CPU> &a);
};

matrix<CPU> operator*(float val, const matrix<CPU>& a);
matrix<CPU> operator+(float val, const matrix<CPU>& a);
matrix<CPU> operator-(float val, const matrix<CPU>& a);
