#pragma once
#include "matrix.h"
#include <vector>
#include <iostream>
#include <numeric>

template<>
class matrix<CUDA> 
{
private:
    float* data;
    size_t r,c, h;
    size_t n;

    static constexpr int THREADS_1D = 256;
    static constexpr int THREADS_2D = 16;

public:

    matrix();
    matrix<CUDA>(const matrix<CUDA>& other);
    matrix<CUDA>(matrix<CUDA>&& other) noexcept;

    matrix<CUDA>(const size_t rows, const size_t columns);
    matrix<CUDA>(const size_t rows, const size_t columns, const std::vector<float>& values);
    matrix<CUDA>(const size_t rows, const size_t columns, float val);
    matrix<CUDA>(const size_t rows, const size_t columns, float start, float end);
    ~matrix<CUDA>();

    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val);
    static matrix<CUDA> create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end);
    static matrix<CUDA> create_stacked_matrix(matrix<CUDA>* begin, matrix<CUDA>* end);

    static matrix<CUDA> add_mat_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b);


    matrix<CUDA> operator%(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+(const float &a) const;
    matrix<CUDA> operator-(const matrix<CUDA> &a) const;
    matrix<CUDA> operator-(const float &a) const;
    matrix<CUDA> operator*(const float &a) const;
    matrix<CUDA> operator*(const matrix<CUDA> &a) const;
    matrix<CUDA> operator+=(const matrix<CUDA> &a);
    matrix<CUDA> operator-=(const matrix<CUDA> &a);

    matrix<CUDA> transpose(const matrix<CUDA> &a);


     
    size_t rows() const;
    size_t columns() const;
    size_t height() const;
    size_t size() const;
    size_t mat_size() const;

    bool empty() const;
    float* raw();


    std::vector<float> sum(); 
    std::vector<float> L2();
    std::vector<size_t> argmax();
    std::vector<size_t> argmin();

    void print();
    void print_size();
    void set(float val);

    static matrix<CUDA> sqrt(const matrix<CUDA> &a);
    static matrix<CUDA> square(const matrix<CUDA> &a);
    static matrix<CUDA> reciprocal(const matrix<CUDA> &a);



};

matrix<CUDA> operator*(float val, const matrix<CUDA>& a);
matrix<CUDA> operator+(float val, const matrix<CUDA>& a);
matrix<CUDA> operator-(float val, const matrix<CUDA>& a);

