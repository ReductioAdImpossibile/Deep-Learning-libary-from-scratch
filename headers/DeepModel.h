#pragma once
#include "backend.h"
#include "matrix.h"
#include "vector"


template<typename Backend>
class model;

template<typename Backend>
class dataset;

template<typename Backend>
class neuralnetwork;

#ifdef ENABLE_CUDA
    #include "modelCUDA.cuh"

#else
    #include "modelCPU.h"
    using NeuralNetwork = neuralnetwork<CPU>;
    using Dataset = dataset<CPU>;
#endif








