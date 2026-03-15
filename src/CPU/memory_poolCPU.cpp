#include "matrixCPU.h"

memory_pool<CPU>& memory_pool<CPU>::instance()
{
    static memory_pool<CPU> pool;
    return pool;
}

float* memory_pool<CPU>::allocate(size_t n)
{
    if(n == 0)
        return nullptr;

    float* ptr = nullptr;
    #pragma omp critical(memory_pool)
    {
        auto& blocks = free_blocks[n];
        if(!blocks.empty())
        {
            ptr = blocks.back();
            blocks.pop_back();
        }
    }

    return ptr ? ptr : new float[n];
}

void memory_pool<CPU>::deallocate(float* ptr, size_t n) 
{
    
    auto& blocks = free_blocks[n];
    if(blocks.size() < 32) 
        blocks.push_back(ptr);
    else
        delete[] ptr; 
}

memory_pool<CPU>::~memory_pool<CPU>() 
{
    for (auto& [size, blocks] : free_blocks)
        for (float* ptr : blocks)
            delete[] ptr;
    free_blocks.clear();
}
