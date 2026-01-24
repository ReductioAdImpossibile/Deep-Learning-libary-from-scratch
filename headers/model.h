#pragma once
#include "backend.h"


template<typename Backend>
class model;

#ifdef ENABLE_CUDA
    #include "tensorCUDA.cu"
    using Model = model<CUDA>;
#else
    #include "tensorCPU.h"
    using Model = model<CPU>;
#endif