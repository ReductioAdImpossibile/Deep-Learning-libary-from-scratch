#include "activationCUDA.cuh"
#include "kernel.cuh"
#include <cmath>

matrix<CUDA> activation<CUDA>::ones(const matrix<CUDA> &a)
{
    matrix<CUDA> result = a;
    result.set(1);
    return a;
}


matrix<CUDA> activation<CUDA>::identity(const matrix<CUDA> &a)
{
    matrix<CUDA> result = a;
    return result;
}

matrix<CUDA> activation<CUDA>::relu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    

       
    return result;
}

matrix<CUDA> activation<CUDA>::elu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    
    return result;
}

matrix<CUDA> activation<CUDA>::sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    
    
    return result;
}

matrix<CUDA> activation<CUDA>::log_sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    

    
    return result;
}

matrix<CUDA> activation<CUDA>::hard_sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    
    return result;
}

matrix<CUDA> activation<CUDA>::tanh(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());

    return result;
}

matrix<CUDA> activation<CUDA>::softmax(const matrix<CUDA> &a)
{
    matrix<CUDA> result = a;


    return result;
}

matrix<CUDA> activation<CUDA>::didentity(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns(), 1);
    return result;
}

matrix<CUDA> activation<CUDA>::drelu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    return result;
}

matrix<CUDA> activation<CUDA>::delu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    
    return result;
}

matrix<CUDA> activation<CUDA>::dsigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> sig = sigmoid(a);
    return sig % (1 - sig);
}

matrix<CUDA> activation<CUDA>::dlog_sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> sig = sigmoid(a);
    return 1 - sig;
}

matrix<CUDA> activation<CUDA>::dhard_sigmoid(const matrix<CUDA> &a)
{

    matrix<CUDA> result(a.rows(), a.columns());
    return result;
}

matrix<CUDA> activation<CUDA>::dtanh(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    return result;
}

activation_fn activation<CUDA>::get_fn(activation_type atype)
{
    switch (atype)
    {
        case activation<CUDA>::IDENTITY:
            return activation<CUDA>::identity;

        case activation<CUDA>::RELU:
            return activation<CUDA>::relu;

        case activation<CUDA>::ELU:
            return activation<CUDA>::elu;

        case activation<CUDA>::SIGMOID:
            return activation<CUDA>::sigmoid;

        case activation<CUDA>::LOG_SIGMOID:
            return activation<CUDA>::log_sigmoid;

        case activation<CUDA>::HARD_SIGMOID:
            return activation<CUDA>::hard_sigmoid;

        case activation<CUDA>::TANH:
            return activation<CUDA>::tanh;

        case activation<CUDA>::SOFTMAX:
            return activation<CUDA>::softmax;

        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

activation_fn activation<CUDA>::get_derivative_fn(activation_type atype)
{
    switch (atype)
    {
        case activation<CUDA>::IDENTITY:
            return activation<CUDA>::didentity;

        case activation<CUDA>::RELU:
            return activation<CUDA>::drelu;

        case activation<CUDA>::ELU:
            return activation<CUDA>::delu;

        case activation<CUDA>::SIGMOID:
            return activation<CUDA>::dsigmoid;

        case activation<CUDA>::LOG_SIGMOID:
            return activation<CUDA>::dlog_sigmoid;

        case activation<CUDA>::HARD_SIGMOID:
            return activation<CUDA>::dhard_sigmoid;

        case activation<CUDA>::TANH:
            return activation<CUDA>::dtanh;

        case activation<CUDA>::SOFTMAX:
            return activation<CUDA>::didentity;

        default:
            throw std::invalid_argument("Unknown activation type");
    }
}




matrix<CUDA> loss<CUDA>::cross_entropy(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    float sum = 0;
    float epsilon = 1e-15f; 

    return matrix<CUDA>();
}

matrix<CUDA> loss<CUDA>::quadratic(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    float sum = 0;
    matrix<CUDA> err_sq = matrix<CUDA>::square(probability - expected);

    return  matrix<CUDA>(); //(err_sq % this->weights).sum() / expected.size();
}


matrix<CUDA> loss<CUDA>::dcross_entropy(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> grad(probability.rows(), probability.columns(), 0);



    return grad;
}

matrix<CUDA> loss<CUDA>::dcross_entropy_inkl_softmax(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    /*
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);
    for (int i = 0; i < expected.size(); i++)
        grad[i] = probability[i] - expected[i];
    */
    
    
    matrix<CUDA> grad = (probability - expected) % this->weights;
    return grad;
}

matrix<CUDA> loss<CUDA>::dquadratic(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    /*
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);

    for (int i = 0; i < expected.size(); i++)
        grad[i] = 2 * (probability[i] - expected[i]);
    */
    matrix<CUDA> grad = 2 * ((probability - expected) % this->weights); 
    return grad;
}




loss<CUDA>::loss()
{
}

loss_fn loss<CUDA>::get_fn(loss_type ltype)
{
    switch (ltype)
    {
        case loss<CUDA>::CROSS_ENTROPY:
            return [this](const matrix<CUDA>& p, const matrix<CUDA>& e) {
                return this->cross_entropy(p, e);
            };

        case loss<CUDA>::QUADRATIC:
            return [this](const matrix<CUDA>& p, const matrix<CUDA>& e) {
                return this->quadratic(p, e);
            };

        default:
            throw std::invalid_argument("Unknown loss type");
    }
}

loss_derivative_fn loss<CUDA>::get_derivative_fn(loss_type ltype, activation_type atype)
{
    switch (ltype)
    {
        case loss<CUDA>::CROSS_ENTROPY:
        {
            if (atype == activation<CUDA>::SOFTMAX)
                return [this](const matrix<CUDA>& p, const matrix<CUDA>& e) {
                    return this->dcross_entropy_inkl_softmax(p, e);
                };

            else
                return [this](const matrix<CUDA>& p, const matrix<CUDA>& e) {
                    return this->dcross_entropy(p, e);
                };
        }

        case loss<CUDA>::QUADRATIC:
                return [this](const matrix<CUDA>& p, const matrix<CUDA>& e) {
                    return this->dquadratic(p, e);
                };

        default:
            throw std::invalid_argument("Unknown loss type");
    }
}
