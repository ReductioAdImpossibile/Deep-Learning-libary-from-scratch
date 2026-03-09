#pragma once
#include "backend.h"
#include "matrix.h"
#include "vector"


template<typename Backend>
class model;

template<typename Backend>
class neuralnetwork;

#ifdef ENABLE_CUDA
    #include "modelCUDA.cuh"

#else
    #include "modelCPU.h"
    using NeuralNetwork = neuralnetwork<CPU>;
    using Dataset = dataset;
#endif



class dataset
{

public:
    std::vector<matrix<CPU>> input;
    std::vector<matrix<CPU>> expected;
    
    dataset();
    dataset(const std::string filename, size_t label_col = 0);
    dataset(const std::string filename, const std::vector<size_t>& ignore, size_t label_col = 0);

    dataset split(float ratio);


    void one_hot_encode();   
    void normalize ();
    void standardize();


    void print_information();

};






