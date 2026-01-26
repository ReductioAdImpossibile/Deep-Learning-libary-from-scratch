#pragma once
#include "backend.h"

template<typename Backend>
class activation;

template<typename Backend>
class loss;

#ifdef ENABLE_CUDA
    #include "activiationCUDA.cu"
    using Activation = activation<CUDA>;
    using Loss = loss<CUDA>;
#else
    #include "activationCPU.h"
    using Activation = activation<CPU>;
    using Loss = loss<CPU>;
#endif
