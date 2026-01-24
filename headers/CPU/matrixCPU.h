#pragma once
#include "tensorCPU.h"
#include "matrix.h"



template<>
class matrix<CPU> : public tensor<CPU>
{
protected:
    size_t c, r;

public:
    matrix<CPU>();
    matrix<CPU>(const std::vector<size_t> &shape);
    matrix<CPU>(const std::vector<size_t> &shape, float val);
    matrix<CPU>(const std::vector<size_t> &shape, float start, float end);
    
    static void mat_mul(const matrix& a, const matrix &b, matrix& result);
    static void mat_mul_transposed(const matrix& a, const matrix &b, matrix& result);
    
    void transpose();
    static void transpose(const matrix &a, matrix &result);

    matrix<CPU> operator*(const matrix<CPU> &a) const;
    
    size_t rows() const;
    size_t columns() const;
    void print() override;


    static matrix<CPU> ReLU(matrix<CPU>& a);
    static matrix<CPU> ELU(matrix<CPU>& a);
    static matrix<CPU> Sigmoid(matrix<CPU>& a);
    static matrix<CPU> Log_Sigmoid(matrix<CPU>& a);
    static matrix<CPU> Hard_Sigmoid(matrix<CPU>& a);
    static matrix<CPU> Tanh(matrix<CPU>& a);
    static matrix<CPU> Hard_Tanh(matrix<CPU>& a);
    static matrix<CPU> Softmax(matrix<CPU>& a);
    
};