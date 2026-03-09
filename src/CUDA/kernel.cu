
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

