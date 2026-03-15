#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>
#include <unordered_map>

template<>
class matrix<CUDA> 
{
private:
    float* data;
    size_t r,c, h;
    size_t n;

    bool owns_memory = true;
    
public:

    static constexpr int THREADS_1D = 256;
    static constexpr int THREADS_2D = 16;

    matrix();
    matrix<CUDA>(const matrix<CUDA>& other);
    matrix<CUDA>(matrix<CUDA>&& other) noexcept;
    matrix<CUDA>(const size_t rows, const size_t columns);
    matrix<CUDA>(const size_t rows, const size_t columns, const std::vector<float>& values);
    matrix<CUDA>(const size_t rows, const size_t columns, float val);
    matrix<CUDA>(const size_t rows, const size_t columns, float start, float end);
    matrix<CUDA>(float* ptr, const size_t rows, const size_t columns, const size_t height);
    ~matrix<CUDA>();

    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end);

    matrix<CUDA> slice_stacked_matrix(size_t start, size_t end);

    static matrix<CUDA> reduce_sum(const matrix<CUDA> &a);
    static matrix<CUDA> bcast_add_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);
    static matrix<CUDA> bcast_hadamard_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);
    static matrix<CUDA> bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);
    static matrix<CUDA> bcast_mat_mul_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);
    static matrix<CUDA> bcast_scale_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);


    float operator[](size_t index) const;
    void set(size_t index, float val);

    matrix<CUDA>& operator=(const matrix<CUDA>& other);
    matrix<CUDA>& operator=(matrix<CUDA>&& other) noexcept;

    matrix<CUDA> operator%(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const float &a) const;
    matrix<CUDA> operator-(const matrix<CUDA> &a) const;
    matrix<CUDA> operator-(const float &a) const;
    matrix<CUDA> operator*(const float &a) const;
    matrix<CUDA> operator*(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+=(const matrix<CUDA> &a);
    matrix<CUDA> operator-=(const matrix<CUDA> &a);

    static matrix<CUDA> transpose(const matrix<CUDA> &a);


     
    size_t rows() const;
    size_t columns() const;
    size_t height() const;
    size_t elements() const;
    size_t mat_elements() const;

    bool empty() const;
    float* raw();
    float* raw() const;
    std::vector<float> values();

    matrix<CUDA> min() const;
    matrix<CUDA> max() const;
    matrix<CUDA> sum() const; 
    matrix<CUDA> L2() const;
    
    std::vector<size_t> argmax() const;
    std::vector<size_t> argmin() const;


    void print() const;
    void print_shape() const;
    void set(float val);

    static matrix<CUDA> sqrt(const matrix<CUDA> &a);
    static matrix<CUDA> square(const matrix<CUDA> &a);
    static matrix<CUDA> reciprocal(const matrix<CUDA> &a);
    static matrix<CUDA> exp(const matrix<CUDA> &a);
    static matrix<CUDA> log2(const matrix<CUDA> &a);



};

matrix<CUDA> operator*(float val, const matrix<CUDA>& a);
matrix<CUDA> operator+(float val, const matrix<CUDA>& a);
matrix<CUDA> operator-(float val, const matrix<CUDA>& a);


template<>
class memory_pool<CUDA>
{
private:
    std::unordered_map<size_t, std::vector<float*>> free_blocks;

public:
    static memory_pool<CUDA>& instance();
    float* allocate(size_t n);
    void deallocate(float* ptr, size_t n);
    ~memory_pool<CUDA>();
};
