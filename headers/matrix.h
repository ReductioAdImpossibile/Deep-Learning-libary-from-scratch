/**
 * @file matrix.h
 * @brief Enables the platform independent interface for the linear algebra libary.
 * 
 *  The libary lets the user work with stacked matrices. All operations work for these. 
 *  It chooses automatically between the CPU and CUDA implementation for the matrix calculations.
 *  Always use 'Matrix' instead of matrix<CPU> or matrix<CUDA>.
 */


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
