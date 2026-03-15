#include "matrixCUDA.cuh"
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "kernel.cuh"


matrix<CUDA>::matrix() : n(0), h(0), c(0), r(0), data(nullptr), owns_memory(false)
{
}

matrix<CUDA>::matrix(const matrix<CUDA>& other) : r(other.r), c(other.c), n(other.n), h(other.h)
{
    //cudaMalloc(&this->data, sizeof(float) * n);
    this->data = memory_pool<CUDA>::instance().allocate(n);
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
    //cudaMalloc(&this->data, sizeof(float) * n);
    this->data = memory_pool<CUDA>::instance().allocate(n);
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

matrix<CUDA>::matrix(float *ptr, const size_t rows, const size_t columns, const size_t height) : data(ptr), r(rows), c(columns), h(height), owns_memory(false)
{
    this->n = rows * columns * height;
}

matrix<CUDA>::~matrix()
{
    if(owns_memory)
        memory_pool<CUDA>::instance().deallocate(data, n);
}



matrix<CUDA> matrix<CUDA>::create_stacked_matrix(const size_t rows, const size_t columns, const size_t height)
{
    matrix<CUDA> result;
    result.r = rows;
    result.c = columns;
    result.h = height;
    result.n = rows * columns * height;
    result.owns_memory = true;

    result.data = memory_pool<CUDA>::instance().allocate(result.n);
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



matrix<CUDA> matrix<CUDA>::slice_stacked_matrix(size_t start, size_t end)
{   
    if(this->h < end)
        throw std::runtime_error("slice_stacked_matrix : matrix height to small.");

    return matrix<CUDA>(this->data + start * mat_elements(), r,c, end - start);
}

// ------------------------------------------

matrix<CUDA> matrix<CUDA>::reduce_sum(const matrix<CUDA> &a)
{
    size_t threads = std::min((size_t)THREADS_1D, a.height());

    matrix<CUDA> result = create_stacked_matrix(a.rows(), a.columns(), 1);
    size_t mat_size = a.mat_elements();
    size_t shared = THREADS_1D * sizeof(float);


    matrix_kernel_reduce_sum<<<mat_size, threads, shared>>>(
        a.data, result.data, mat_size, a.height());
    return result;
}

matrix<CUDA> matrix<CUDA>::bcast_add_to_stacked_matrix(const matrix<CUDA> &a, const matrix<CUDA> &b)
{   

    size_t mat_size = a.mat_elements();

    if(b.height() != 1 || a.rows() != b.rows() || a.columns() != b.columns() )
        throw std::runtime_error("bcast_add_to_stacked_matrix : matrix dimensions must be equal and the height of the second matrix needs to be 1.");    
    
    dim3 threads(256);
    dim3 blocks(
        (mat_size + 255) / 256,             // round div by 256 up
        a.h
    );

    matrix<CUDA> result = a;
    matrix_kernel_bcast_add_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, mat_size, a.elements());
    return result;
}

matrix<CUDA> matrix<CUDA>::bcast_hadamard_to_stacked_matrix(const matrix<CUDA> &a, const matrix<CUDA> &b)
{
    size_t mat_size = a.mat_elements();
    if(b.height() != 1 || a.rows() != b.rows() || a.columns() != b.columns() )
        throw std::runtime_error("bcast_hadamard_to_stacked_matrix : height needs to be equal to 1");    
    
    dim3 threads(256);
    dim3 blocks(
        (mat_size + 255) / 256,            
        a.h
    );

    matrix<CUDA> result = a;
    matrix_kernel_bcast_hadamard_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, mat_size, a.elements());
    return result;
}

matrix<CUDA> matrix<CUDA>::bcast_reversed_mat_mul_to_stacked_matrix(const matrix<CUDA> &a, const matrix<CUDA> &b)
{

    
    if(b.columns() != a.rows() || b.height() != 1)
        throw std::runtime_error("bcast_reversed_mat_mul_to_stacked_matrix : matrix shapes do not match. b.columns() needs to be equal to a.rows() (reversed mat mul!) ");

    matrix<CUDA> result = create_stacked_matrix(b.rows(), a.columns(), a.height());

    dim3 threads(16, 16);
    dim3 blocks(
        (result.rows() + 15) / 16,
        (result.columns() + 15) / 16,
        result.height()
    );

    matrix_kernel_bcast_reversed_mat_mul_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, result.r, result.c, b.columns());

    return result;
}

matrix<CUDA> matrix<CUDA>::bcast_mat_mul_to_stacked_matrix(const matrix<CUDA> &a, const matrix<CUDA> &b)
{
    if(a.columns() != b.rows() || b.height() != 1)
        throw std::runtime_error("bcast_mat_mul_to_stacked_matrix : matrix shapes do not match. a.columns() needs to be equal to b.rows()");

    matrix<CUDA> result = create_stacked_matrix(b.rows(), a.columns(), a.height());

    dim3 threads(16, 16);
    dim3 blocks(
        (result.rows() + 15) / 16,
        (result.columns() + 15) / 16,
        result.height()
    );

    matrix_kernel_bcast_mat_mul_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, result.r, result.c, b.columns());
    return result;

}

matrix<CUDA> matrix<CUDA>::bcast_scale_to_stacked_matrix(const matrix<CUDA> &a, const matrix<CUDA> &b)
{
    size_t mat_size = a.mat_elements();
    if(b.rows() != 1 || b.columns() != 1 )
        throw std::runtime_error("bcast_scale_to_stacked_matrix : Height of matrices need to match and second matrix has to be of shape 1x1xh");    
    
    dim3 threads(256);
    dim3 blocks(
        (mat_size + 255) / 256,             // round div by 256 up
        a.h
    );

    matrix<CUDA> result = a;
    matrix_kernel_bcast_scale_to_stacked_matrix<<<blocks, threads>>>(a.data, b.data, result.data, mat_size, a.elements());
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

size_t  matrix<CUDA>::elements() const
{
    return n;
}

size_t matrix<CUDA>::mat_elements() const
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

float *matrix<CUDA>::raw() const
{
    return data;
}

std::vector<float> matrix<CUDA>::values()
{
    std::vector<float> result(this->n);
    cudaMemcpy(result.data(), this->data, this->n *sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

matrix<CUDA> matrix<CUDA>::sum() const
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* block_sums;
    cudaMalloc(&block_sums, this->h * amount_blocks * sizeof(float));


    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_sum<<<blocks, THREADS_1D, shared>>>(this->data, block_sums, mat_size);

    matrix<CUDA> result = create_stacked_matrix(1,1, this->h);
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_sum<<<blocks2, THREADS_1D, shared>>>(block_sums, result.data, amount_blocks);
    }
    else 
        cudaMemcpy(result.data, block_sums, this->h * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(block_sums);
    return result;
}

matrix<CUDA> matrix<CUDA>::L2() const
{
    matrix<CUDA> squared = matrix<CUDA>::square(*this);
    
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* block_sums;
    cudaMalloc(&block_sums, this->h * amount_blocks * sizeof(float));

    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_sum<<<blocks, THREADS_1D, shared>>>(squared.data, block_sums, mat_size);

    matrix<CUDA> result = create_stacked_matrix(1,1,this->h);
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_sum<<<blocks2, THREADS_1D, shared>>>(block_sums, result.data, amount_blocks);
    }
    else 
        cudaMemcpy(result.data, block_sums, this->h * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(block_sums);

    return result;
}

matrix<CUDA> matrix<CUDA>::max() const
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* max_values;
    cudaMalloc(&max_values, this->h * amount_blocks * sizeof(float));


    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_max<<<blocks, THREADS_1D, shared>>>(this->data, max_values, mat_size);

    matrix<CUDA> result = create_stacked_matrix(1,1, this->h);
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_max<<<blocks2, THREADS_1D, shared>>>(max_values, result.data, amount_blocks);
    }
    else 
        cudaMemcpy(result.data, max_values, this->h * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(max_values);
    return result;
}

matrix<CUDA> matrix<CUDA>::min() const
{
    size_t mat_size = this->r * this->c;
    size_t amount_blocks = (mat_size + THREADS_1D - 1 ) / THREADS_1D;
    size_t shared = THREADS_1D * sizeof(float);

    float* min_values;
    cudaMalloc(&min_values, this->h * amount_blocks * sizeof(float));


    dim3 blocks(amount_blocks, this->h);
    matrix_kernel_min<<<blocks, THREADS_1D, shared>>>(this->data, min_values, mat_size);

    matrix<CUDA> result = create_stacked_matrix(1,1, this->h);
    if(amount_blocks > 1)
    {
        dim3 blocks2(1, this->h);
        matrix_kernel_min<<<blocks2, THREADS_1D, shared>>>(min_values, result.data, amount_blocks);
    }
    else 
        cudaMemcpy(result.data, min_values, this->h * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(min_values);
    return result;
}





std::vector<size_t>  matrix<CUDA>::argmax() const
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

std::vector<size_t>   matrix<CUDA>::argmin() const
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


void  matrix<CUDA>::print() const 
{
    std::vector<float> arr(n);
    cudaMemcpy(arr.data(), this->data, sizeof(float) * n, cudaMemcpyDeviceToHost);



    for(int l = 0; l < this->h; l++)
    {
        std::cout << "----------- Matrix : " << l << " -----------" << std::endl;
        size_t offset = l * mat_elements();
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

void  matrix<CUDA>::print_shape() const
{
    std::cout << r << " " << c << " " << h << std::endl;
}

void  matrix<CUDA>::set(float val)
{

    if(val == 0.0f)
    {
        cudaMemset(this->data, 0, n * sizeof(float));  
        return;
    }

    size_t blocks = (this->n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_set<<<blocks, THREADS_1D>>>(this->data, val, this->n);
}

matrix<CUDA>  matrix<CUDA>::sqrt(const matrix<CUDA> &a)
{
    
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_sqrt<<<blocks, THREADS_1D>>>(a.data, res.data, res.n);
    return res;
}

matrix<CUDA>  matrix<CUDA>::square(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_square<<<blocks, THREADS_1D>>>(a.data, res.data, res.n);
    return res;
}

matrix<CUDA>  matrix<CUDA>::reciprocal(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_reciprocal<<<blocks, THREADS_1D>>>(a.data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::exp(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_exp<<<blocks, THREADS_1D>>>(a.data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::log2(const matrix<CUDA> &a)
{
    matrix<CUDA> res = a;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_log2<<<blocks, THREADS_1D>>>(a.data, res.data, res.n);
    return res;
}




float matrix<CUDA>::operator[](size_t index) const
{
    float v;
    cudaMemcpy(&v, &data[index], sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

void matrix<CUDA>::set(size_t index, float val)
{
    cudaMemcpy(&data[index], &val, sizeof(float), cudaMemcpyHostToDevice);
}

matrix<CUDA>& matrix<CUDA>::operator=(const matrix<CUDA>& other)
{
    if (this != &other) {
        if (data && owns_memory) 
            memory_pool<CUDA>::instance().deallocate(data,n); 
        r = other.r;
        c = other.c;
        n = other.n;
        h = other.h;
        owns_memory = true;
        data = memory_pool<CUDA>::instance().allocate(n);
        cudaMemcpy(data, other.data, n * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

matrix<CUDA>& matrix<CUDA>::operator=(matrix<CUDA>&& other) noexcept
{
    if (this != &other) 
    {
        if (data && owns_memory) 
            memory_pool<CUDA>::instance().deallocate(data,n); 
        r = other.r;
        c = other.c;
        h = other.h;
        n = other.n;
        owns_memory = other.owns_memory;
        data = other.data;      

        
        other.data = nullptr;      
        other.r = other.c = other.n = other.h = 0;
    }
    return *this;
}



matrix<CUDA> matrix<CUDA>::operator%(const matrix<CUDA> &a) const
{
    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("% : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_hadamard_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_hadamard_to_stacked_matrix(a, *this);

    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_hadamard<<<blocks, THREADS_1D>>>(a.data, this->data, res.data, res.n);
    return res;
}

matrix<CUDA> matrix<CUDA>::operator+(const matrix<CUDA> &a) const
{  

    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("+ : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_add_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_add_to_stacked_matrix(a, *this);

    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_add<<<blocks, THREADS_1D>>>(this->data, a.data, res.data, res.n);
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
    if(this->r != a.rows() || this->c != a.columns() || (a.height() > 1 && this->h > 1 && a.height() != this->h) )
        throw std::runtime_error("- : matrix shapes need to be equal ");

    if(this->h > 1 && a.height() == 1)
        return bcast_add_to_stacked_matrix(*this, a * (-1));

    if(this->h == 1 && a.height() > 1)
        return bcast_add_to_stacked_matrix(a * (-1), *this);

    matrix<CUDA> res = *this;
    size_t blocks = (res.n + THREADS_1D - 1) / THREADS_1D;

    matrix_kernel_sub<<<blocks, THREADS_1D>>>(this->data, a.data , res.data, res.n);
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
    if(this->c != a.rows())
        throw std::runtime_error("mat_mul : matrix shapes do not match. ");

    if(this->h > 1 && a.height() == 1)
        return bcast_mat_mul_to_stacked_matrix(*this, a);

    if(this->h == 1 && a.height() > 1)
        return bcast_reversed_mat_mul_to_stacked_matrix(a, *this);


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
    *this = *this + a;
    return *this;
}

matrix<CUDA> matrix<CUDA>::operator-=(const matrix<CUDA> &a)
{
    *this = *this - a;
    return *this;
}

matrix<CUDA> matrix<CUDA>::transpose(const matrix<CUDA> &a)
{
    matrix<CUDA> result = create_stacked_matrix(a.columns(), a.rows(), a.height());
    dim3 blocks((a.mat_elements() + 255) / 256, a.height());

    matrix_kernel_transpose<<<blocks, 256>>>(a.data, result.data, result.rows(), result.columns(), result.elements());
    return result;
}









