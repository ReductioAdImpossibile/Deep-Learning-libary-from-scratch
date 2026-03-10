#include "matrixCUDA.cuh"
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "kernel.cuh"


matrix<CUDA>::matrix() : n(0), h(0), c(0), r(0)
{
}

matrix<CUDA>::matrix(const matrix<CUDA>& other) : r(other.r), c(other.c), n(other.n), h(other.h)
{
    cudaMalloc(&this->data, sizeof(float) * n);
    cudaMemcpy(this->data, other.data, sizeof(float) * other.n, cudaMemcpyDeviceToDevice);
}

matrix<CUDA>::matrix(matrix<CUDA>&& other) noexcept : n(other.n), r(other.r), c(other.c), h(other.h), data(other.data)
{
    other.n = 0;
    other.r = 0;
    other.c = 0;
    other.h = 0;
    other.data = nullptr;
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
    cudaMemcpy(this->data, values.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
}

matrix<CUDA>::matrix(const size_t rows, const size_t columns, float val) : matrix(rows, columns)
{
    std::vector<float> arr(n, val);
    cudaMemcpy(this->data, arr.data(), sizeof(float) * n ,cudaMemcpyHostToDevice);
}

matrix<CUDA>::matrix(const size_t rows, const size_t columns, float start, float end) : matrix(rows, columns)
{
    std::vector<float> arr(n);
    std::mt19937 gen(
        std::random_device{}()
    );
    std::uniform_real_distribution<float> dist(start, end);

    for (size_t i = 0; i < this->n; ++i){
        arr[i] = dist(gen);
    }
    cudaMemcpy(this->data, arr.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

}

matrix<CUDA>::~matrix()
{
    cudaFree(this->data);
}



matrix<CUDA> matrix<CUDA>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height)
{
    matrix<CUDA> result;
    result.r = rows;
    result.c = columns;
    result.h = height;
    result.n = rows * columns * height;

    cudaMalloc(&result.data, sizeof(float) * result.n);
    return result;
}

matrix<CUDA> matrix<CUDA>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, const std::vector<float>& values)
{
    if(values.size() != rows * columns * height)
        throw std::runtime_error("create_stacked_matrix : vector size and mat dim's dont match.");

    matrix<CUDA> result = create_stacked_matrix(rows, columns, height);
    cudaMemcpy(result.data, values.data(), result.n * sizeof(float), cudaMemcpyHostToDevice);
    return result;
}

matrix<CUDA> matrix<CUDA>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float val)
{   

    matrix<CUDA> result = create_stacked_matrix(rows, columns, height);
    std::vector<float> arr(result.n, val);

    cudaMemcpy(result.data, arr.data(), result.n * sizeof(float),  cudaMemcpyHostToDevice);
    return result;
}

matrix<CUDA> matrix<CUDA>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height, float start, float end)
{
    matrix<CUDA> result = create_stacked_matrix(rows, columns, height);
    std::vector<float> arr(result.n);
    
    std::mt19937 gen(
        std::random_device{}()
    );
    std::uniform_real_distribution<float> dist(start, end);

    for (size_t i = 0; i < result.n; ++i)
    {
        arr[i] = dist(gen);
    }
    cudaMemcpy(result.data, arr.data(), sizeof(float) * result.n, cudaMemcpyHostToDevice);
    return result;
}

matrix<CUDA> matrix<CUDA>::create_stacked_matrix(matrix<CUDA>* begin, matrix<CUDA>* end)
{
    size_t height = end - begin;
    size_t size = begin->size();
    matrix<CUDA> result = create_stacked_matrix(begin->rows(), begin->columns(), height);

    matrix<CUDA>* mat = begin;
    for(size_t i = 0; i < height; i++ )
    {
        if(mat->size() != size)
            throw std::runtime_error("create_stacked_matrix : Matrices inside the array need to have the same size.");

        cudaMemcpy(result.data + i * size , mat->data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        mat++;
    }
    return result;
}


// ------------------------------------------


matrix<CUDA> matrix<CUDA>::add_mat_to_stacked_matrix(const matrix<CUDA>& a, const matrix<CUDA>& b)
{   

    size_t mat_size = a.mat_size();

    if(mat_size != b.mat_size() || b.height() != 1)
        throw std::runtime_error("add_mat_to_stacked_matrix : matrix dimensions must be equal and the height of the second matrix needs to be 1.");    
    
    dim3 threads(256);
    dim3 blocks(
        (mat_size + 255) / 256,             // round div by 256 up
        a.h
    );

    matrix<CUDA> result = a;
    matrix_kernel_add_mat_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, mat_size, a.size());
    return result;
}







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

size_t matrix<CUDA>::mat_size() const
{
    return this->r * this->c;
}

bool  matrix<CUDA>::empty() const
{
    return (this->n==0);
}

float* matrix<CUDA>::raw()
{
    return data;
}


std::vector<float> matrix<CUDA>::sum()
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* block_sums;
    cudaMalloc(&block_sums, this->h * amount_blocks * sizeof(float));

    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_sum<<<blocks, THREADS_1D, shared>>>(this->data, block_sums, mat_size);

    
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_sum<<<blocks2, THREADS_1D, shared>>>(block_sums, block_sums, amount_blocks);
    }
    
    std::vector<float> result(this->h);
    cudaMemcpy(result.data(), block_sums, this->h * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(block_sums);

    return result;
} 

std::vector<float> matrix<CUDA>::L2()
{
    
    matrix<CUDA> squared = matrix<CUDA>::square(*this);
    
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* block_sums;
    cudaMalloc(&block_sums, this->h * amount_blocks * sizeof(float));

    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_sum<<<blocks, THREADS_1D, shared>>>(squared.data, block_sums, mat_size);

    
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_sum<<<blocks2, THREADS_1D, shared>>>(block_sums, block_sums, amount_blocks);
    }
    
    std::vector<float> result(this->h);
    cudaMemcpy(result.data(), block_sums, this->h * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(block_sums);

    return result;
}

std::vector<size_t>  matrix<CUDA>::argmax()
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(size_t);

    
    size_t* max_indices;
    cudaMalloc(&max_indices, this->h * amount_blocks * sizeof(size_t));

    
    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_argmax<<<blocks, THREADS_1D, shared>>>(this->data, max_indices, mat_size);

    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_argmax<<<blocks2, THREADS_1D, shared>>>(this->data, max_indices, amount_blocks);
    }

    std::vector<size_t> result(this->h);

    cudaMemcpy(result.data(), max_indices, this->h * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(max_indices);
    return result;
}

std::vector<size_t>   matrix<CUDA>::argmin()
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(size_t);

    
    size_t* min_indices;
    cudaMalloc(&min_indices, this->h * amount_blocks * sizeof(size_t));

    
    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_argmin<<<blocks, THREADS_1D, shared>>>(this->data, min_indices, mat_size);

    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_argmin<<<blocks2, THREADS_1D, shared>>>(this->data, min_indices, amount_blocks);
    }

    std::vector<size_t> result(this->h);

    cudaMemcpy(result.data(), min_indices, this->h * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(min_indices);

    return result;
}

void  matrix<CUDA>::print()
{
    std::vector<float> arr(n);
    cudaMemcpy(arr.data(), this->data, sizeof(float) * n, cudaMemcpyDeviceToHost);


    for(int l = 0; l < this->h; l++)
    {
        std::cout << "----------- Matrix : " << l << " -----------" << std::endl;
        size_t offset = l * this->h;
        for(size_t row = 0; row < this->r; row++)
        {
            for(size_t col = 0; col< this->c; col++)
            {
                std::cout << arr[row * this->c + col + offset] << " ";
            }
            std::cout << std::endl;
        }
    }

}

void  matrix<CUDA>::print_size()
{
    std::cout << r << " " << c << " " << h << std::endl;
}

void  matrix<CUDA>::set(float val)
{
    std::vector<float> arr(n, val);
    cudaMemcpy(this->data, arr.data(), sizeof(float) * n , cudaMemcpyHostToDevice);
}

matrix<CUDA>  matrix<CUDA>::sqrt(const matrix<CUDA> &a)
{
    
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_sqrt<<<blocks, THREADS_1D>>>(res.data, res.n);
    return res;
}

matrix<CUDA>  matrix<CUDA>::square(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_square<<<blocks, THREADS_1D>>>(res.data, res.n);
    return res;
}

matrix<CUDA>  matrix<CUDA>::reciprocal(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_reciprocal<<<blocks, THREADS_1D>>>(res.data, res.n);
    return res;
}




matrix<CUDA> matrix<CUDA>::operator%(const matrix<CUDA> &a) const
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_hadamard<<<blocks, THREADS_1D>>>(a.data, this->data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator+(const matrix<CUDA> &a) const
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_add<<<blocks, THREADS_1D>>>(a.data, this->data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator+(const float &a) const
{
    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_add_value<<<blocks, THREADS_1D>>>(this->data, res.data, a, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator-(const matrix<CUDA> &a) const
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_sub<<<blocks, THREADS_1D>>>(a.data, this->data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator-(const float &a) const
{
    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_add_value<<<blocks, THREADS_1D>>>(this->data, res.data, -a, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator*(const float &a) const
{
    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_scale<<<blocks, THREADS_1D>>>(this->data, res.data, a, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator*(const matrix<CUDA> &a) const
{
    if(this->h != a.height())
        throw std::runtime_error("mat_mul : matrices stacks arent equal in size. ");
    
    if(this->c != a.rows())
        throw std::runtime_error("mat_mul : matrix shapes do not match. ");

    matrix<CUDA> result = create_stacked_matrix(this->r, a.columns(), this->h);

    dim3 threads(16, 16);
    dim3 blocks(
        (result.rows() + 15) / 16,
        (result.columns() + 15) / 16,
        result.height()
    );

    matrix_kernel_mat_mul<<<blocks, threads>>>(
        this->data, a.data, result.data, result.rows(), result.columns(), a.rows());

    return result;
}

matrix<CUDA> operator*(float val, const matrix<CUDA> &a)
{
    return a * val;
}

matrix<CUDA> operator+(float val, const matrix<CUDA> &a)
{
    return a + val;
}

matrix<CUDA> operator-(float val, const matrix<CUDA> &a)
{
    return (-1) * a + val;
}



matrix<CUDA> matrix<CUDA>::operator+=(const matrix<CUDA> &a)
{
    matrix<CUDA> res = *this + a;
    return res;
}

matrix<CUDA> matrix<CUDA>::operator-=(const matrix<CUDA> &a)
{
    matrix<CUDA> res = *this - a;
    return res;
}

matrix<CUDA> matrix<CUDA>::transpose(const matrix<CUDA> &a)
{
    matrix<CUDA> result = create_stacked_matrix(a.columns(), a.rows(), a.height());
    dim3 threads(256);
    dim3 blocks((a.mat_size() + 255) / 256, a.height());

    matrix_kernel_transpose<<<blocks, threads>>>(a.data, result.data, result.rows(), result.columns(), result.size());
    return result;
}

