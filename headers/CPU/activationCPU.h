#pragma once
#include "activation.h"
#include "matrixCPU.h"
#include <functional>



template<>
class activation<CPU>
{   
private:
    static matrix<CPU> ones(const matrix<CPU>& a);
    static matrix<CPU> identity(const matrix<CPU>& a);
    static matrix<CPU> relu(const matrix<CPU>& a);
    static matrix<CPU> elu(const matrix<CPU>& a);
    static matrix<CPU> sigmoid(const matrix<CPU>& a);
    static matrix<CPU> log_sigmoid(const matrix<CPU>& a);
    static matrix<CPU> hard_sigmoid(const matrix<CPU>& a);
    static matrix<CPU> tanh(const matrix<CPU>& a);
    static matrix<CPU> softmax(const matrix<CPU>& a);


    static matrix<CPU> didentity(const matrix<CPU>& a);
    static matrix<CPU> drelu(const matrix<CPU>& a);
    static matrix<CPU> delu(const matrix<CPU>& a);
    static matrix<CPU> dsigmoid(const matrix<CPU>& a);
    static matrix<CPU> dlog_sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dhard_sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dtanh(const matrix<CPU>& a);

   inline static float ELU_ALPHA_PARAM = 1;

public:

    static activation_type IDENTITY     = 0;
    static activation_type RELU         = 1;
    static activation_type ELU          = 2;
    static activation_type SIGMOID      = 3;
    static activation_type LOG_SIGMOID  = 4;
    static activation_type HARD_SIGMOID = 5;
    static activation_type TANH         = 6;
    static activation_type SOFTMAX      = 7;

    static activation_fn get_fn(activation_type atype);
    static activation_fn get_derivative_fn(activation_type atype);
};



template<>
class loss<CPU>
{  
private:
    
    float cross_entropy(const matrix<CPU> &expected, const matrix<CPU> &result);
    float quadratic(const matrix<CPU> &expected, const matrix<CPU> &result);

    matrix<CPU> dcross_entropy(const matrix<CPU> &probability, const matrix<CPU> &expected);
    matrix<CPU> dcross_entropy_inkl_softmax(const matrix<CPU> &probability, const matrix<CPU> &expected);
    matrix<CPU> dquadratic(const matrix<CPU> &probability, const matrix<CPU> &expected);

    
    

public:

    matrix<CPU> weights;
    loss<CPU>();
    static loss_type CROSS_ENTROPY = 0;
    static loss_type QUADRATIC = 1;

    loss_fn get_fn(loss_type ltype);
    loss_derivative_fn get_derivative_fn(loss_type ltype, activation_type atype);
};


template<>
class optimizer<CPU>
{
public:
    static optimizer_type STOCHASTIC_GRADIENT_DESCENT = 0;
    static optimizer_type BATCH_GRADIENT_DESCENT = 1;
    static optimizer_type MIN_BATCH_GRADIENT_DESCENT = 2;
};




template<>
class adam_optimizer<CPU> : private optimizer<CPU>
{
public:
    adam_optimizer(); 
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double lambda;
    size_t batch_size;
};

inline adam_optimizer<CPU>::adam_optimizer() : lr(0.001), beta1(0.9), beta2(0.999), epsilon(1e-8), lambda(10e-4), batch_size(64)
{}



template<>
class hyperparameter<CPU> : private optimizer<CPU>
{
public:
    hyperparameter(); 
    double lr;
    double lambda;
    size_t batch_size;
};

inline hyperparameter<CPU>::hyperparameter() : lr(0.001), lambda(10e-4), batch_size(64)
{}

