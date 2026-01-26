#include "matrixCPU.h"
#include <random>
#include <omp.h>
#include <stdexcept>
#include <string>

matrix<CPU>::matrix()
{
    
}

matrix<CPU>::matrix(const matrix<CPU> &other) : c(other.c), r(other.r), shape(other.shape), n(other.n)
{
    this->data = (float*) _mm_malloc(n * sizeof(float), ALIGN);
    std::copy(other.data, other.data + n, data);
}

matrix<CPU>::matrix(const std::vector<size_t> &_shape) :shape(_shape), r(_shape[0]), c(_shape[1])
{
    this->n = r * c;
    this->data = (float*) _mm_malloc(n * sizeof(float), ALIGN);
    if (!data) throw std::bad_alloc();
}

matrix<CPU>::matrix(const std::vector<size_t> &shape, float val) : matrix<CPU>(shape)
{
    const size_t limit = (n/w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v(val);
        v.copy_to(this->data + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < n; i++)
        this->data[i] = val;
}

matrix<CPU>::matrix(const std::vector<size_t> &shape, float start, float end) : matrix<CPU>(shape)
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

matrix<CPU>::~matrix()
{
    if(data) _mm_free(data);
}

std::vector<size_t> matrix<CPU>::get_shape() const
{
    return this->shape;
}

size_t matrix<CPU>::rows() const
{
    return this->r;
}

size_t matrix<CPU>::columns() const
{
    return this->c;
}

size_t matrix<CPU>::size() const
{
    return this->n;
}

float *matrix<CPU>::raw()
{
    return this->data;
}

float *matrix<CPU>::raw() const
{
    return this->data;
}


// --------------------------------- OPERATOR ------------------------------
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
    matrix<CPU> result(this->shape);
    matrix<CPU>::hadamard(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->shape);
    matrix<CPU>::add(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const float &a) const
{
    matrix<CPU> result(this->shape);
    
    const ssize_t limit = (n / w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v;    
        v.copy_from(data + i, std::experimental::element_aligned);
        v = v + a;
        v.copy_to(result.raw() + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < this->n; i++)
        result[i] = data[i] + a;

    return result;
}

matrix<CPU> matrix<CPU>::operator-(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->shape);
    matrix<CPU>::sub(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator-(const float &a) const
{
    matrix<CPU> result(this->shape);
    
    const ssize_t limit = (n / w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v;    
        v.copy_from(data + i, std::experimental::element_aligned);
        v = v - a;
        v.copy_to(result.raw() + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < this->n; i++)
        result[i] = data[i] - a;

    return result;
}

matrix<CPU> matrix<CPU>::operator*(const float &a) const
{
    matrix<CPU> result(this->shape);
    matrix<CPU>::scale(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator*(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->shape);
    matrix<CPU>::mat_mul(*this, a, result) ;
    return result;
}





void matrix<CPU>::mat_mul(const matrix &a, const matrix &b, matrix &result)
{

    if(a.columns() != b.rows() || result.columns() != b.columns() || result.rows() != a.rows())
        throw std::runtime_error("matmul : matrix shapes do not match.");

    matrix<CPU> transposed_b({b.columns(), b.rows()});
    matrix<CPU>::transpose(b, transposed_b);

    size_t cols = result.columns();
    size_t rows = result.rows();


    #pragma omp parallel for schedule(static)
    for(size_t row = 0; row < rows; row++)
    {
        for(size_t col = 0; col < cols; col++)
        {
            float sum = 0;
            fsimd sum_simd(0);
            fsimd _a, _b;
            
            float* a_pos = a.raw() + row * a.columns();
            float* b_pos = transposed_b.raw() + col * a.columns();
            size_t i = 0;

            for(; i + w <= a.columns() ; i += w)
            {
                
                _a.copy_from(a_pos + i, stdx::element_aligned);
                _b.copy_from(b_pos + i, stdx::element_aligned);

                sum_simd += _a * _b;
            }         
            
            for(; i < a.columns(); i++)
                sum += a_pos[i] * b_pos[i];

            sum += stdx::reduce(sum_simd, std::plus{});   

            result[row * cols + col] = sum;
        }
    }

    
}

void matrix<CPU>::mat_mul_transposed(const matrix& a, const matrix &b, matrix& result)
{

    /*
    Transposes b bevore doing matrix multiplication.
     => A * B^T 
    */ 

    if(a.columns() != b.rows() || result.columns() != b.columns() || result.rows() != a.rows())
        throw std::runtime_error("matmul : matrix shapes do not match.");

    size_t cols = result.columns();
    size_t rows = result.rows();
    
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            float sum = 0;
            for(int i = 0; i < a.columns(); i++)
                sum += a[i] * b[i];
            result[row * cols + col] = sum;
        }
    }
}

void matrix<CPU>::transpose(const matrix& a, matrix& result)
{

    if(a.columns() != result.rows() && a.rows() != result.columns())
        throw std::runtime_error("transpose : matrix shapes do not match.");

    size_t columns = a.columns();
    size_t rows = a.rows();
    const float* data = a.raw();

    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col< columns; col++)
        {
            result[col * result.columns() + row] = data[row * columns + col];
        }
    }
}

void matrix<CPU>::transpose()
{
    matrix<CPU> result({this->columns(),this->rows()});
    transpose(*this, result);
    free(data);
    *this = result;
}

void matrix<CPU>::print()
{
    
    for(int row = 0; row < r; row++)
    {
        for(int col = 0; col< c; col++)
        {
            std::cout << this->data[row * c + col] << " ";
        }
        std::cout << std::endl;
    }
}

void matrix<CPU>::set(float val)
{
    const ssize_t limit = (n / w) * w;
    #pragma omp parallel for
    for(int i = 0; i < limit; i += w)
    {
        fsimd v(val);
        v.copy_to(data + i, std::experimental::element_aligned);
    }

    for(int i = limit; i < this->n; i++)
        data[i] = val;
}





bool matrix<CPU>::equal_shape(const matrix<CPU> &a, const matrix<CPU> &b)
{
    return a.get_shape() == b.get_shape();
}

void matrix<CPU>::hadamard(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the hadamard product. They need to be the same");
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();

    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(size_t i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ * b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = limit ;j < n; j++)
        result_raw[j] = a_raw[j] * b_raw[j];
}

void matrix<CPU>::add(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the tensor addition. They need to be the same");
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit ; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ + b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = (n/w) * w ;j < n; j++)
        result_raw[j] = a_raw[j] + b_raw[j];

}

void matrix<CPU>::sub(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( !(a.get_shape() == b.get_shape() && b.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the tensor addition. They need to be the same");
    
    const float* a_raw = a.raw();
    const float* b_raw = b.raw();   
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;
    

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);
            b_.copy_from(b_raw + i, std::experimental::element_aligned);
        
            result_ = a_ - b_;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = limit ;j < n; j++)
        result_raw[j] = a_raw[j] - b_raw[j];


}

void matrix<CPU>::scale(const matrix<CPU> &a, const float value, matrix<CPU> &result)
{
     if( !(a.get_shape() == result.get_shape()) )
        throw std::runtime_error("Tensor shapes do not match for the hadamard product. They need to be the same");
    
    
    const float* a_raw = a.raw();  
    float* result_raw = result.raw();
    const size_t n = a.size();
    const size_t limit = (n/w) * w;

    #pragma omp parallel
    {
        fsimd a_, b_, result_;
        #pragma omp for
        for(int i = 0; i < limit; i += w)
        {
            a_.copy_from(a_raw + i, std::experimental::element_aligned);    
            result_ = a_ * value;
            result_.copy_to(result_raw + i, std::experimental::element_aligned);
        }

    }
    for(int j = (n/w) * w ;j < n; j++)
        result_raw[j] = a_raw[j] * value;

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
    return a - val;
}
