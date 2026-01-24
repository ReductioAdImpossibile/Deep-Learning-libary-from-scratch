#include "tensorCPU.h"
#include "matrixCPU.h"




matrix<CPU>::matrix(const std::vector<size_t> &shape) : tensor(shape)
{
    if(shape.size() > 2 )
        throw std::runtime_error("Matrix shape needs to be 2 dimensional.");
    
    if(shape.size() == 2)
        this->c = shape[1];
    else
        this->c = 1;

    this->r = shape[0];
}

matrix<CPU>::matrix(const std::vector<size_t> &shape, float val) : tensor(shape, val), c(shape[1]), r(shape[0])
{
    if(shape.size() > 2 )
        throw std::runtime_error("Matrix shape needs to be 2 dimensional.");
    
    if(shape.size() == 2)
        this->c = shape[1];
    else
        this->c = 1;

    this->r = shape[0];
}

matrix<CPU>::matrix(const std::vector<size_t> &shape, float start, float end) : tensor(shape, start, end)
{
    if(shape.size() > 2 )
        throw std::runtime_error("Matrix shape needs to be 2 dimensional.");
    
    if(shape.size() == 2)
        this->c = shape[1];
    else
        this->c = 1;

    this->r = shape[0];
}

matrix<CPU> matrix<CPU>::operator*(const matrix<CPU> &a) const
{
    return matrix<CPU>();
}

size_t matrix<CPU>::rows() const
{
    return this->r;
}

size_t matrix<CPU>::columns() const
{
    return this->c;
}

void matrix<CPU>::mat_mul(const matrix& a, const matrix &b, matrix& result)
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