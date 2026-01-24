#pragma once
#include "backend.h"


template<typename Backend>
class vector;

#ifdef ENABLE_CUDA
    #include "vectorCUDA.cu"
    using Vector = vector<CUDA>;
#else
    #include "vectorCPU.h"
    using Vector = vector<CPU>;
#endif
