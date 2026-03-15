#pragma once
#include "backend.h"


template<typename Backend>
class matrix;

template<typename Backend>
class memory_pool;

#ifdef ENABLE_CUDA
    #include "matrixCUDA.cuh"
    using Matrix = matrix<CUDA>;
    using Memory_Pool = memory_pool<CUDA>;
#else
    #include "matrixCPU.h"
    using Matrix = matrix<CPU>;
    using Memory_Pool = memory_pool<CPU>;
#endif
