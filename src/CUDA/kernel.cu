#include "kernel.cuh"



__global__ void matrix_kernel_sum(const float* data, float *result, const size_t mat_size)
{
    extern __shared__ float shared_data[];
    
    size_t threadid = threadIdx.x;
    size_t layer = blockIdx.y;

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = layer * mat_size;

    shared_data[threadid] = (index < mat_size) ? data[offset + index] : 0;
    __syncthreads();


    for(size_t stride = blockDim.x / 2;  stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            shared_data[threadid] += shared_data[threadid + stride];
        }
        __syncthreads();
    }

    if(threadid == 0)
        result[layer * gridDim.x + blockIdx.x] = shared_data[0];
}

__global__ void matrix_kernel_argmax(const float* data, size_t* result, const size_t mat_size)
{
    extern __shared__ size_t shared_indices[];

    size_t threadid = threadIdx.x;
    size_t layer = blockIdx.y;

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = layer * mat_size;

    shared_indices[threadid] = (index < mat_size) ? offset + index : 0;

    __syncthreads();


    for(size_t stride = blockDim.x / 2;  stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            
            if(data[shared_indices[threadid]] < data[shared_indices[threadid + stride]])
            {
                shared_indices[threadid] = shared_indices[threadid + stride];
            }
        }
        __syncthreads();
    }

    if(threadid == 0)
        result[layer * gridDim.x + blockIdx.x] = shared_indices[0];
    
        
}

__global__ void matrix_kernel_argmin(const float* data, size_t* result, const size_t mat_size)
{
    extern __shared__ size_t shared_indices[];

    size_t threadid = threadIdx.x;
    size_t layer = blockIdx.y;

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = layer * mat_size;

    shared_indices[threadid] = (index < mat_size) ? offset + index : 0;

    __syncthreads();


    for(size_t stride = blockDim.x / 2;  stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            
            if(data[shared_indices[threadid]] > data[shared_indices[threadid + stride]])
            {
                shared_indices[threadid] = shared_indices[threadid + stride];
            }
        }
        __syncthreads();
    }

    if(threadid == 0)
        result[layer * gridDim.x + blockIdx.x] = shared_indices[0];
    
        
}

__global__ void matrix_kernel_sqrt(float* data, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        data[i] = std::sqrt(data[i]);

}

__global__ void matrix_kernel_square(float* data, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        data[i] = data[i] * data[i];
}

__global__ void matrix_kernel_reciprocal(float* data, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        data[i] = 1 / data[i];
        
}

__global__ void matrix_kernel_hadamard(const float *A, const float *B, float *result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] * B[i];
}

__global__ void matrix_kernel_add(const float *A, const float *B, float *result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] * B[i];
}

__global__ void matrix_kernel_sub(const float *A,const float *B, float *result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] * B[i];
}

__global__ void matrix_kernel_scale(const float *A, float *result, const float value,  const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] * value;
}

__global__ void matrix_kernel_add_value(const float *A, float *result, const float value, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] + value;
}

__global__ void matrix_kernel_mat_mul(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t column = blockIdx.y * blockDim.y + threadIdx.y;
    size_t layer = blockIdx.z;

    size_t offset_a = layer * (result_rows * length);
    size_t offset_b = layer * (length * result_cols);
    size_t offset_result = layer * (result_rows * result_cols);

    if(row >= result_rows || column >= result_cols) return;

    float sum = 0;
    for(size_t i = 0; i < length; i++)
    {
        sum += A[offset_a + row * length + i] * B[offset_b + column + i * result_cols ]; 
    }
    result[offset_result + row * result_cols + column] = sum;
}

__global__ void matrix_kernel_add_mat_to_stacked_matrix(const float *A, const float *B, float *result, const size_t mat_size, const size_t n)
{       
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.y * mat_size;

    if(i >= mat_size || offset + i >= n)
        return;

    result[offset + i] = A[offset + i] + B[i]; 
}

__global__ void matrix_kernel_transpose(const float* A, float* result, const size_t result_rows, const size_t result_columns, const size_t n)
{
    size_t mat_size = result_rows * result_columns;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.y * (result_rows * result_columns);

    if(i + offset >= n || i >= mat_size)
        return;

    size_t row = i / result_columns;
    size_t col = i % result_rows;

    result[offset + col * result_rows + row] = A[offset + i];

}












