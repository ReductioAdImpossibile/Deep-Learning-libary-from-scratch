#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <unordered_map>

template<>
class matrix<CPU> 
{
private:
    float* data;
    size_t r,c, h;
    size_t n;
    bool owns_memory = true;
    
public:

    matrix();
    matrix<CPU>(const matrix<CPU>& other);
    matrix<CPU>(matrix<CPU>&& other) noexcept;
    matrix<CPU>(const size_t rows, const size_t columns);
    matrix<CPU>(const size_t rows, const size_t columns, const std::vector<float>& values);
    matrix<CPU>(const size_t rows, const size_t columns, float val);
    matrix<CPU>(const size_t rows, const size_t columns, float start, float end);
    matrix<CPU>(float* ptr, const size_t rows, const size_t columns, const size_t height);
    ~matrix<CPU>();

    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height);
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values);
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val);
    static matrix<CPU> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end);

    matrix<CPU> slice_stacked_matrix(size_t start, size_t end);

    static matrix<CPU> reduce_sum(const matrix<CPU> &a);
    static matrix<CPU> bcast_add_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);
    static matrix<CPU> bcast_hadamard_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);
    static matrix<CPU> bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);
    static matrix<CPU> bcast_mat_mul_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);
    static matrix<CPU> bcast_scale_to_stacked_matrix(const matrix<CPU>& a, const matrix<CPU>& b);


    const float operator[](size_t index) const;
    float& operator[](size_t index);
    void set(size_t index, float val);

    matrix<CPU>& operator=(const matrix<CPU>& other);
    matrix<CPU>& operator=(matrix<CPU>&& other) noexcept;

    matrix<CPU> operator%(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const matrix<CPU> &a) const;
    matrix<CPU> operator+(const float &a) const;
    matrix<CPU> operator-(const matrix<CPU> &a) const;
    matrix<CPU> operator-(const float &a) const;
    matrix<CPU> operator*(const float &a) const;
    matrix<CPU> operator*(const matrix<CPU> &a) const;
    matrix<CPU> operator+=(const matrix<CPU> &a);
    matrix<CPU> operator-=(const matrix<CPU> &a);

    static matrix<CPU> transpose(const matrix<CPU> &a);


     
    size_t rows() const;
    size_t columns() const;
    size_t height() const;
    size_t elements() const;
    size_t mat_elements() const;

    bool empty() const;
    float* raw();
    float* raw() const;
    std::vector<float> values();

    matrix<CPU> min() const;
    matrix<CPU> max() const;
    matrix<CPU> sum() const; 
    matrix<CPU> L2() const;
    
    std::vector<size_t> argmax() const;
    std::vector<size_t> argmin() const;


    void print() const;
    void print_shape() const;
    void set(float val);

    static matrix<CPU> sqrt(const matrix<CPU> &a);
    static matrix<CPU> square(const matrix<CPU> &a);
    static matrix<CPU> reciprocal(const matrix<CPU> &a);
    static matrix<CPU> exp(const matrix<CPU> &a);
    static matrix<CPU> log2(const matrix<CPU> &a);

};

matrix<CPU> operator*(float val, const matrix<CPU>& a);
matrix<CPU> operator+(float val, const matrix<CPU>& a);
matrix<CPU> operator-(float val, const matrix<CPU>& a);


template<>
class memory_pool<CPU>
{
private:
    std::unordered_map<size_t, std::vector<float*>> free_blocks;

public:
    static memory_pool<CPU>& instance();
    float* allocate(size_t n);
    void deallocate(float* ptr, size_t n);
    ~memory_pool<CPU>();
};
