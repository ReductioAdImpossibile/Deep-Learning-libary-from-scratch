#pragma once
#include "backend.h"
#include "matrix.h"
#include <cstddef>
#include <functional>

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


using activation_type = const size_t;
using loss_type = const size_t;
using loss_derivative_type = const size_t;
using optimizer_type = const size_t;


#ifdef ENABLE_CUDA

    using activation_fn = std::function<matrix<CUDA>(const matrix<CUDA>&)>;
    using loss_fn = std::function<float(const matrix<CUDA>&, const matrix<CUDA>&)>;
    using loss_derivative_fn = std::function<matrix<CUDA>(const matrix<CUDA>&, const matrix<CUDA>&)>;

    #include "activationCUDA.cuh"
    using Activation = activation<CUDA>;
    using Loss = loss<CUDA>;
    using Optimizer = optimizer<CUDA>;
    using ADAM_Optimizer = adam_optimizer<CUDA>;
    using Hyperparameter = hyperparameter<CUDA>;



#else

    using activation_fn = std::function<matrix<CPU>(const matrix<CPU>&)>;
    using loss_fn = std::function<float(const matrix<CPU>&, const matrix<CPU>& )>;
    using loss_derivative_fn = std::function<matrix<CPU>(const matrix<CPU>&, const matrix<CPU>&)>;

    #include "activationCPU.h"

    using Activation = activation<CPU>;
    using Loss = loss<CPU>;
    using Optimizer = optimizer<CPU>;
    using ADAM_Optimizer = adam_optimizer<CPU>;
    using Hyperparameter = hyperparameter<CPU>;

#endif

