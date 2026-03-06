#pragma once
#include "backend.h"

template<typename Backend>
class activation;

template<typename Backend>
class loss;

template<typename Backend>
class optimizer;

template<typename Backend>
class adam_optimizer;

template<typename Backend>
class hyperparameter;


#ifdef ENABLE_CUDA
    #include "activiationCUDA.cu"
    using Activation = activation<CUDA>;
    using Loss = loss<CUDA>;
#else
    #include "activationCPU.h"
    using Activation = activation<CPU>;
    using Loss = loss<CPU>;
    using Optimizer = optimizer<CPU>;
    using ADAM_Optimizer = adam_optimizer<CPU>;
    using Hyperparameter = hyperparameter<CPU>;
#endif
