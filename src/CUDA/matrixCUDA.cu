#include "matrixCUDA.cuh"
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>



matrix<CUDA>::matrix() : n(0), h(0), c(0), r(0)
{
    
}

matrix<CUDA>::matrix(const matrix<CUDA>& other)
{

}

matrix<CUDA>::matrix(matrix<CUDA>&& other)
{

}




matrix<CUDA>::matrix(const size_t rows, const size_t columns) : r(rows), c(columns), h(1)
{
    this->n = r * c;
    cudaMalloc(&this->data, sizeof(float) * n);
}

matrix<CUDA>::matrix(const size_t rows, const size_t columns, const std::vector<float>& values) : matrix(rows, columns)
{

    if(values.size() != n)
    {
        throw std::runtime_error("Matrix : Size of matrix does not equal the input vector size");
    }
    cudaMemcpy(&this->data, values.data(), sizeof(float) * n ,cudaMemcpyHostToDevice);
}

matrix<CUDA>::matrix(const size_t rows, const size_t columns, float val) : matrix(rows, columns)
{
    float arr[this->n];
    memset(arr, val, n * sizeof(float));
    cudaMemcpy(&this->data, &arr, sizeof(float) * n ,cudaMemcpyHostToDevice);
}

matrix<CUDA>::matrix(const size_t rows, const size_t columns, float start, float end)
{
    float arr[this->n];
    
    std::mt19937 gen(
        std::random_device{}()
    );
    std::uniform_real_distribution<float> dist(start, end);

    for (size_t i = 0; i < this->n; ++i)
    {
        arr[i] = dist(gen);
    }
    cudaMemcpy(&this->data, &arr, sizeof(float) * n ,cudaMemcpyHostToDevice);

}

matrix<CUDA>::~matrix()
{
    cudaFree(this->data);
}










// ------------------------------------------



size_t matrix<CUDA>::rows() const
{
    return r;
}


size_t  matrix<CUDA>::columns() const
{
    return c;
}

size_t  matrix<CUDA>::height() const
{
    return h;
}

size_t  matrix<CUDA>::size() const
{
    return n;
}

bool  matrix<CUDA>::empty() const
{
    return (n==0);
}

double  matrix<CUDA>::sum()
{
    
}

double  matrix<CUDA>::L1()
{

}

double  matrix<CUDA>::L2()
{

}

size_t  matrix<CUDA>::argmax()
{

}

size_t  matrix<CUDA>::argmin()
{

}

void  matrix<CUDA>::print()
{

}

void  matrix<CUDA>::print_size()
{
    std::cout << r << " " << c << " " << h << std::endl;
}

void  matrix<CUDA>::set(float val)
{
    float arr[this->n];
    memset(arr, val, n * sizeof(float));
    cudaMemcpy(&this->data, &arr, sizeof(float) * n ,cudaMemcpyHostToDevice);
}

matrix<CUDA>  matrix<CUDA>::sqrt(const matrix<CUDA> &a)
{

}

matrix<CUDA>  matrix<CUDA>::square(const matrix<CUDA> &a)
{

}

matrix<CUDA>  matrix<CUDA>::reciprocal(const matrix<CUDA> &a)
{

}