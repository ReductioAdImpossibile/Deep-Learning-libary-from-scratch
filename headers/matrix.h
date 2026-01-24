#pragma once
#include "backend.h"


template<typename Backend>
class matrix;

#ifdef ENABLE_CUDA
    #include "matrixCUDA.cu"
    using Matrix = matrix<CUDA>;
#else
    #include "matrixCPU.h"
    using Matrix = matrix<CPU>;
#endif
