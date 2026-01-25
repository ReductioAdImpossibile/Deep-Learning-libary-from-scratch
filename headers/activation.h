#pragma once
#include "backend.h"

template<typename Backend>
class activation;

#ifdef ENABLE_CUDA
    #include "activiationCUDA.cu"
    using Activation = activation<CUDA>;
#else
    #include "activationCPU.h"
    using Activation = activation<CPU>;
#endif
