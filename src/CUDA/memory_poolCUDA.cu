
#include "matrixCUDA.cuh"





memory_pool<CUDA>& memory_pool<CUDA>::instance()
{
    static memory_pool<CUDA> pool;
    return pool;
}

float* memory_pool<CUDA>::allocate(size_t n)
{
    auto& blocks = free_blocks[n];
    if(!blocks.empty())
    {
        float* ptr = blocks.back();
        blocks.pop_back();
        return ptr;
    }
    float *ptr;
    cudaMalloc(&ptr, sizeof(float) * n);
    return ptr;
}

void memory_pool<CUDA>::deallocate(float* ptr, size_t n) 
{
    if(ptr == nullptr) return;
        free_blocks[n].push_back(ptr);
}

memory_pool<CUDA>::~memory_pool<CUDA>() 
{
    for (auto& [size, blocks] : free_blocks)
        for (float* ptr : blocks)
            cudaFree(ptr);
    free_blocks.clear();
}


