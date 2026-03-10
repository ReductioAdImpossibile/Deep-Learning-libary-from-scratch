#include "matrixCPU.h"
#include <random>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <algorithm>



matrix<CPU>::matrix()
    : r(0), c(0), h(0), n(0)
{}


matrix<CPU>::matrix(const matrix<CPU> &other) 
    : r(other.r), c(other.c), n(other.n), h(other.h), data(other.data)
{}


matrix<CPU>::matrix(matrix<CPU> &&other) noexcept
    : r(other.r), c(other.c), h(other.h), n(other.n), data(std::move(other.data))
{
    other.n = 0;
    other.r = 0;
    other.c = 0;
    other.h = 0;
}

matrix<CPU>::matrix(const size_t rows, const size_t columns) :r(rows), c(columns), h(1)
{
    this->n = r * c * 1;
    this->data.resize(n);
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, const std::vector<float> &values) : r(rows), c(columns), h(1)
{
    n = r * c;
    if(values.size() != n)
        throw std::runtime_error("Vector size does not match matrix size");

    this->data = values;
}


matrix<CPU>::matrix(const size_t rows, const size_t columns, float val) : matrix<CPU>(rows, columns)
{
    for(int i = 0; i < this->n; i++)
        this->data[i] = val;
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, float start, float end) : matrix<CPU>(rows, columns)
{
    #pragma omp parallel
    {
        std::mt19937 gen(
            std::random_device{}() + omp_get_thread_num()
        );
        std::uniform_real_distribution<float> dist(start, end);

        #pragma omp for
        for (size_t i = 0; i < this->n; ++i)
        {
            this->data[i] = dist(gen);
        }
    }
}


matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height)
{
    matrix<CPU> res;
    res.r = rows;
    res.c = columns; 
    res.h = height;
    res.n = height * columns * rows;
    return res; 
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float> &values)
{
    matrix<CPU> res = create_stacked_matrix(rows, columns, height);
    res.data = values;
    return res;
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val)
{
    matrix<CPU> res = create_stacked_matrix(rows, columns, height);
    for(size_t i = 0; i < res.n; i++)
        res[i] = val;
    return res;
}

matrix<CPU> matrix<CPU>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end)
{
    matrix<CPU> res = create_stacked_matrix(rows, columns, height);
    #pragma omp parallel
    {
        std::mt19937 gen(
            std::random_device{}() + omp_get_thread_num()
        );
        std::uniform_real_distribution<float> dist(start, end);

        #pragma omp for
        for (size_t i = 0; i < res.n; ++i)
        {
            res[i] = dist(gen);
        }
    }
    return res;

}







size_t matrix<CPU>::rows() const
{
    return this->r;
}

size_t matrix<CPU>::columns() const
{
    return this->c;
}

size_t matrix<CPU>::height() const
{
    return this->h;
}

size_t matrix<CPU>::size() const
{
    return this->n;
}

bool matrix<CPU>::empty() const
{
    return data.empty();
}

std::vector<float> &matrix<CPU>::raw()
{
    return this->data;
}

std::vector<float> matrix<CPU>::raw_copy()
{
    return this->data;
}

double matrix<CPU>::sum()
{
    double res = 0;
    for(size_t i = 0; i < n; i++)
        res += data[i];
    return static_cast<float>(res);
}

double matrix<CPU>::L1()
{
    double res = 0;
    for(size_t i = 0; i < n; i++)
        res += std::abs(data[i]);
    return static_cast<float>(res);
}

double matrix<CPU>::L2()
{
    double res = 0;
    for(size_t i = 0; i < n; i++)
        res += data[i] * data[i];
    return static_cast<float>(std::sqrt(res));
}

size_t matrix<CPU>::argmax()
{
    auto max = max_element(data.begin(), data.end());
    return distance(data.begin(), max); 
}

size_t matrix<CPU>::argmin()
{
    auto max = min_element(data.begin(), data.end());
    return distance(data.begin(), max); 
}

double matrix<CPU>::max()
{
    return *std::max_element(data.begin(), data.end());
}

double matrix<CPU>::min()
{
    return *std::min_element(data.begin(), data.end());
}

matrix<CPU> &matrix<CPU>::operator=(const matrix<CPU> &other)
{
    if (this != &other) {
        r = other.r;
        c = other.c;
        n = other.n;
        h = other.h;
        data = other.data;  
    }
    return *this;
}

matrix<CPU>& matrix<CPU>::operator=(matrix<CPU>&& other) noexcept
{
    if (this != &other) {
        r = other.r;
        c = other.c;
        h = other.h;
        n = other.n;
        data  = std::move(other.data);  
        
        other.r = other.c = other.n = 0;
    }
    return *this;
}

const float &matrix<CPU>::operator[](size_t index) const
{
    return this->data[index];
}

float &matrix<CPU>::operator[](size_t index)
{
    return this->data[index];
}

matrix<CPU> matrix<CPU>::operator%(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    
    matrix<CPU>::hadamard(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::add(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    for(int i = 0; i < n; i++)
        result[i] = this->data[i] +  a; 
    return result;
}

matrix<CPU> matrix<CPU>::operator-(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::sub(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator-(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    for(int i = 0; i < n; i++)
        result[i] = this->data[i] - a; 
    return result;
}

matrix<CPU> matrix<CPU>::operator*(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::scale(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator*(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, a.columns()); 
    matrix<CPU>::mat_mul(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+=(const matrix<CPU> &a) 
{
    matrix<CPU>::add(*this, a, *this);
    return *this;
}

matrix<CPU> matrix<CPU>::operator-=(const matrix<CPU> &a) 
{
    matrix<CPU>::sub(*this, a, *this);
    return *this;
}

void matrix<CPU>::mat_mul(const matrix &a, const matrix &b, matrix &result)
{

    if(a.columns() != b.rows() || result.columns() != b.columns() || result.rows() != a.rows() || b.height() != a.height() || result.height() != a.height())
        throw std::runtime_error("matmul : matrix shapes do not match.");

    const size_t rows = a.rows();
    const size_t cols = b.columns();
    const size_t inner = a.columns();
    const size_t layers = a.height();


    const size_t a_offset = rows * inner;
    const size_t b_offset = inner * cols;
    const size_t res_offset = rows * cols;

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t layer = 0; layer < layers; layer++)
        for(size_t row = 0; row < rows; row++) 
        {
            size_t a_start = layer * a_offset;
            size_t b_start = layer * b_offset;
            size_t res_start = layer * res_offset;

            for(size_t col = 0; col < cols; col++)
            {
                float sum = 0;
                for(size_t i = 0; i < inner; i++)
                    sum += a[a_start + row * inner + i] * b[b_start + i * cols + col];
                result[res_start + row * cols + col] = sum;
            }

        }


}

void matrix<CPU>::mat_mul_transposed(const matrix& a, const matrix &b, matrix& result)
{


    if(a.height() != b.height() || a.height() != result.height() || a.columns() != b.columns() || a.rows() != result.rows() || b.rows() != result.columns())
        throw std::runtime_error("matmul : matrix shapes do not match.");

    /*
    Transposes b bevore doing matrix multiplication.
     => A * B^T 
    */ 

    const size_t cols = result.columns();
    const size_t rows = result.rows();
    const size_t layers = a.height();
    const size_t inner = a.columns();

    const size_t a_offset = rows * inner;
    const size_t b_offset = inner * cols;
    const size_t res_offset = rows * cols;

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t layer = 0; layer < layers; layer++)
        for(size_t row = 0; row < rows; row++)
        {
            size_t a_start = layer * a_offset + row * inner;
            size_t res_start = layer * res_offset + row * cols;


            for(size_t col = 0; col < cols; col++)
            {
                size_t b_start = layer * b_offset + col * inner; // we actually loop through the row and not the column!

                float sum = 0;
                for(size_t i = 0; i < inner; i++)
                    sum += a[i + a_start] * b[i + b_start];
                
                result[col + res_start] = sum;
            }


        }
}

void matrix<CPU>::transpose(const matrix& a, matrix& result)
{

    if(a.columns() != result.rows() || a.rows() != result.columns())
        throw std::runtime_error("transpose : matrix shapes do not match.");

    size_t columns = a.columns();
    size_t rows = a.rows();

    for(size_t layer = 0; layer < a.height(); layer++)
    {
        size_t offset = layer * (rows * columns);
        for(size_t row = 0; row < rows; row++)
        {
            for(size_t col = 0; col< columns; col++)
            {
                result[col * result.columns() + row + offset] = a[row * columns + col + offset];
            }
        }
    }

}

matrix<CPU> matrix<CPU>::transpose()
{
    matrix<CPU> result(this->c, this->r);
    transpose(*this, result);
    return result;
}


void matrix<CPU>::print()
{
    
    for(int l = 0; l < this->h; l++)
    {
        std::cout << "----------- Matrix : " << l << " -----------" << std::endl;
        size_t offset = l * this->h;
        for(size_t row = 0; row < this->r; row++)
        {
            for(size_t col = 0; col< this->c; col++)
            {
                std::cout << this->data[row * this->c + col + offset] << " ";
            }
            std::cout << std::endl;
        }
    }
}

void matrix<CPU>::print_size()
{
    std::cout << this->r << " " << this->c << " " << this->h << std::endl;
}

void matrix<CPU>::set(float val)
{
    for(int i = 0; i < n; i++)
        data[i] = val;
}


void matrix<CPU>::hadamard(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{   
    if( a.rows() != b.rows() || 
        b.rows() != result.rows() || 
        a.columns() != b.columns() || 
        b.columns() != result.columns() || 
        a.height() != b.height() || 
        b.height() != result.height()
    )
        throw std::runtime_error("Matrix shapes do not match for the hadamard product. They need to be the same");

    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] * b[i];

}

void matrix<CPU>::add(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( a.rows() != b.rows() || 
        b.rows() != result.rows() || 
        a.columns() != b.columns() || 
        b.columns() != result.columns() || 
        a.height() != b.height() || 
        b.height() != result.height()
    )
        throw std::runtime_error("Matrix shapes do not match for the tensor addition. They need to be the same");
    
    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] + b[i];
}

void matrix<CPU>::sub(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( a.rows() != b.rows() || 
        b.rows() != result.rows() || 
        a.columns() != b.columns() || 
        b.columns() != result.columns() || 
        a.height() != b.height() || 
        b.height() != result.height()
    )
        throw std::runtime_error("Matrix shapes do not match for the tensor substraction. They need to be the same");

    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] - b[i];
}

void matrix<CPU>::scale(const matrix<CPU> &a, const float value, matrix<CPU> &result)
{
    if( a.rows() != result.rows() || a.columns() != result.columns() || result.height() != a.height())
        throw std::runtime_error("Matrix shapes do not match for the tensor scalar. They need to be the same");
            
    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] * value;

}

matrix<CPU> matrix<CPU>::sqrt(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = std::sqrt(a[i]);
    }
    return res;
}

matrix<CPU> matrix<CPU>::square(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = a[i] * a[i];
    }
    return res;
}

matrix<CPU> matrix<CPU>::reciprocal(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = 1 /  a[i];
    }
    return res;
}

matrix<CPU> operator*(float val, const matrix<CPU> &a)
{
    return a * val;
}

matrix<CPU> operator+(float val, const matrix<CPU> &a)
{
    return a + val;
}

matrix<CPU> operator-(float val, const matrix<CPU> &a)
{
    return (-1) * a + val;
}
