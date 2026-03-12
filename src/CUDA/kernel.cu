#include "kernel.cuh"
#include <float.h>


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

__global__ void matrix_kernel_max(const float* data, float* result, const size_t mat_size)
{
    extern __shared__ float shared_data[];
    
    size_t threadid = threadIdx.x;
    size_t layer = blockIdx.y;

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = layer * mat_size;

    shared_data[threadid] = (index < mat_size) ? data[offset + index] : -FLT_MAX;
    __syncthreads();


    for(size_t stride = blockDim.x / 2;  stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            if(shared_data[threadid] < shared_data[threadid + stride])
                shared_data[threadid] = shared_data[threadid + stride];
        }
        __syncthreads();
    }

    if(threadid == 0)
        result[layer * gridDim.x + blockIdx.x] = shared_data[0];
}

__global__ void matrix_kernel_min(const float *data, float *result, const size_t mat_size)
{
        extern __shared__ float shared_data[];
    
    size_t threadid = threadIdx.x;
    size_t layer = blockIdx.y;

    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t offset = layer * mat_size;

    shared_data[threadid] = (index < mat_size) ? data[offset + index] : FLT_MAX;
    __syncthreads();


    for(size_t stride = blockDim.x / 2;  stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            if(shared_data[threadid] > shared_data[threadid + stride])
                shared_data[threadid] = shared_data[threadid + stride];
        }
        __syncthreads();
    }

    if(threadid == 0)
        result[layer * gridDim.x + blockIdx.x] = shared_data[0];

}

__global__ void matrix_kernel_sqrt(const float* data, float* result, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = std::sqrt(data[i]);

}

__global__ void matrix_kernel_square(const float* data, float* result, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = data[i] * data[i];
}

__global__ void matrix_kernel_reciprocal(const float* data, float* result, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = 1 / data[i];
        
}

__global__ void matrix_kernel_exp(const float* data, float* result, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = std::exp(data[i]);
}

__global__ void matrix_kernel_log2(const float* data, float* result, size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = std::log2(data[i]);
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
        result[i] = A[i] + B[i];
}

__global__ void matrix_kernel_sub(const float *A,const float *B, float *result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;

    for(; i < n; i += stride)
        result[i] = A[i] - B[i];
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
 
__global__ void matrix_kernel_bcast_add_to_stacked_matrix(const float *A, const float *B, float *result, const size_t mat_size, const size_t n)
{       
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.y * mat_size;

    if(i >= mat_size || offset + i >= n)
        return;

    result[offset + i] = A[offset + i] + B[i]; 
}

__global__ void matrix_kernel_bcast_reversed_mat_mul_to_stacked_matrix(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t column = blockIdx.y * blockDim.y + threadIdx.y;
    size_t layer = blockIdx.z;

    size_t offset_a = layer * (length * result_cols);
    size_t offset_result = layer * (result_rows * result_cols);

    if(row >= result_rows || column >= result_cols) return;

    float sum = 0;
    for(size_t i = 0; i < length; i++)
    {
        sum += A[offset_a + column + i * result_cols] * B[row * length + i];
    }
    result[offset_result + row * result_cols + column] = sum;
}

__global__ void matrix_kernel_bcast_scale_to_stacked_matrix(const float *A, const float *B, float *result, const size_t mat_size, const size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t layer = blockIdx.y;
    size_t offset = blockIdx.y * mat_size;
    

    if(i >= mat_size || offset + i >= n)
        return;

    result[offset + i] = A[offset + i] * B[layer]; 
}

__global__ void matrix_kernel_transpose(const float* A, float* result, const size_t result_rows, const size_t result_columns, const size_t n)
{
    size_t mat_size = result_rows * result_columns;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = blockIdx.y * (result_rows * result_columns);

    if(i + offset >= n || i >= mat_size)
        return;

    size_t row = i / result_columns;
    size_t col = i % result_columns;

    result[offset + col * result_rows + row] = A[offset + i];

}









__global__ void activation_function_kernel_relu(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = A[i] > 0 ? A[i] : 0;
}

__global__ void activation_function_kernel_elu(const float* A, float* result, const float alpha, const size_t n)
{

    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = A[i] > 0 ? A[i] : alpha * (std::exp(A[i]) - 1);
}

__global__ void activation_function_kernel_sigmoid(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = 1 / (1 + std::exp(-A[i]));
}


__global__ void activation_function_kernel_log_sigmoid(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = std::log(1 / (1 + std::exp(-A[i])));
}


__global__ void activation_function_kernel_hard_sigmoid(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
    {
        float _d = A[i];        
        if(_d < -3)
            result[i] = 0;
        else if(_d > 3)
            result[i] = 1;
        else
            result[i] = _d / 6 + 1/2;

    }
}


__global__ void activation_function_kernel_tanh(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = std::tanh(A[i]);

}


__global__ void activation_function_kernel_drelu(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = A[i] > 0 ? 1 : 0;
}

__global__ void activation_function_kernel_delu(const float* A, float* result, const float alpha, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = A[i] > 0 ? 1 : alpha * std::exp(A[i]) ;
}


__global__ void activation_function_kernel_dhard_sigmoid(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
    {
        float _d = A[i];        
        if(_d < -3 || _d > 3)
            result[i] = 0;
        else
            result[i] = 1 / 6;

    }
}

__global__ void activation_function_kernel_dtanh(const float* A, float* result, const size_t n)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for(; i < n; i += stride)
        result[i] = 1 / (std::cosh(A[i]) * std::cosh(A[i]));
}


