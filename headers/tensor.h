#pragma once
#include "backend.h"


template<typename Backend>
class tensor;

#ifdef ENABLE_CUDA
    #include "tensorCUDA.cu"
    using Tensor = tensor<CUDA>;
#else
    #include "tensorCPU.h"
    using Tensor = tensor<CPU>;
#endif

