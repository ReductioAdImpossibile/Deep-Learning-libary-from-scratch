#include "activationCPU.h"
#include <cmath>





matrix<CPU> activation<CPU>::ones(const matrix<CPU> &a)
{
    matrix<CPU> result = a;
    result.set(1);
    return a;
}


matrix<CPU> activation<CPU>::identity(const matrix<CPU> &a)
{
    matrix<CPU> result = a;
    return result;
}

matrix<CPU> activation<CPU>::relu(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    for(int i = 0;i < a.elements(); i++)
        result[i] = a[i] > 0 ? a[i] : 0; 
       
    return result;
}

matrix<CPU> activation<CPU>::elu(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(int i = 0; i < a.elements(); i++)
        result[i] = a[i] > 0 ? a[i] : ELU_ALPHA_PARAM * (std::exp(a[i]) - 1);

    return result;
}

matrix<CPU> activation<CPU>::sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    

    for(int i = 0; i < a.elements(); i++)
        result[i] = 1 / (1 + std::exp(-a[i]));
    
    return result;
}

matrix<CPU> activation<CPU>::log_sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    for(int i = 0; i < a.elements(); i++)
        result[i] = std::log(1 / (1 + std::exp(-a[i])));
    
    return result;
}

matrix<CPU> activation<CPU>::hard_sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    for(int i = 0; i < a.elements(); i++)
    {
        float _d = a[i];
        
        if(_d < -3)
            result[i] = 0;
        else if(_d > 3)
            result[i] = 1;
        else
            result[i] = _d / 6 + 1/2;
    }
       

    return result;
}

matrix<CPU> activation<CPU>::tanh(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(int i = 0; i < a.elements(); i++)
    {
        result[i] = std::tanh(a[i]); 
    }
       

    return result;
}

matrix<CPU> activation<CPU>::softmax(const matrix<CPU> &a)
{
    matrix<CPU> ones_bcast = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height(), 1);

    matrix<CPU> max_rows = a.max(); 

    matrix<CPU> max_expanded = matrix<CPU>::bcast_scale_to_stacked_matrix(ones_bcast, max_rows);

    matrix<CPU> exp = matrix<CPU>::exp(a - max_expanded);

    matrix<CPU> exp_sum = exp.sum();

    matrix<CPU> exp_sum_expanded = matrix<CPU>::bcast_scale_to_stacked_matrix(ones_bcast, exp_sum);

    return exp % matrix<CPU>::reciprocal(exp_sum_expanded);
}

matrix<CPU> activation<CPU>::didentity(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height(), 1);
    return result;
}

matrix<CPU> activation<CPU>::drelu(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(int i = 0; i < a.elements(); i++)
    {
        float _d = a[i];
        
        if(_d <= 0)
            result[i] = 0;
        else
            result[i] = 1;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::delu(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    for(int i = 0; i < a.elements(); i++)
    {
        float _d = a[i];
        if(_d <= 0)
            result[i] = ELU_ALPHA_PARAM * std::exp(_d);
        else
            result[i] = 1;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dsigmoid(const matrix<CPU> &a)
{
    matrix<CPU> sig = sigmoid(a);
    return sig % (1 - sig);
}

matrix<CPU> activation<CPU>::dlog_sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> sig = sigmoid(a);
    return 1 - sig;
}

matrix<CPU> activation<CPU>::dhard_sigmoid(const matrix<CPU> &a)
{

    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(int i = 0; i < a.elements(); i++)
    {   
        result[i] = a[i] > 3 || a[i] < -3 ? 0 : 1/6;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dtanh(const matrix<CPU> &a)
{
    matrix<CPU> result = matrix<CPU>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    for(int i = 0; i < a.elements(); i++)
    {
        result[i] = 1 / (std::cosh(a[i]) * std::cosh(a[i]));
    }
       

    return result;
}

std::function<matrix<CPU>(const matrix<CPU>&)> activation<CPU>::get_fn(activation_type atype)
{
    switch (atype)
    {
        case activation<CPU>::IDENTITY:
            return activation<CPU>::identity;

        case activation<CPU>::RELU:
            return activation<CPU>::relu;

        case activation<CPU>::ELU:
            return activation<CPU>::elu;

        case activation<CPU>::SIGMOID:
            return activation<CPU>::sigmoid;

        case activation<CPU>::LOG_SIGMOID:
            return activation<CPU>::log_sigmoid;

        case activation<CPU>::HARD_SIGMOID:
            return activation<CPU>::hard_sigmoid;

        case activation<CPU>::TANH:
            return activation<CPU>::tanh;

        case activation<CPU>::SOFTMAX:
            return activation<CPU>::softmax;

        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

std::function<matrix<CPU>(const matrix<CPU>&)> activation<CPU>::get_derivative_fn(activation_type atype)
{
    switch (atype)
    {
        case activation<CPU>::IDENTITY:
            return activation<CPU>::didentity;

        case activation<CPU>::RELU:
            return activation<CPU>::drelu;

        case activation<CPU>::ELU:
            return activation<CPU>::delu;

        case activation<CPU>::SIGMOID:
            return activation<CPU>::dsigmoid;

        case activation<CPU>::LOG_SIGMOID:
            return activation<CPU>::dlog_sigmoid;

        case activation<CPU>::HARD_SIGMOID:
            return activation<CPU>::dhard_sigmoid;

        case activation<CPU>::TANH:
            return activation<CPU>::dtanh;

        case activation<CPU>::SOFTMAX:
            return activation<CPU>::didentity;

        default:
            throw std::invalid_argument("Unknown activation type");
    }
}





float loss<CPU>::cross_entropy(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> prod = matrix<CPU>::log2(probability) % expected;

    matrix<CPU> weighted_prod = matrix<CPU>::bcast_hadamard_to_stacked_matrix(prod, this->weights);

    return matrix<CPU>::reduce_sum(-1 * weighted_prod.sum())[0];
}

float loss<CPU>::quadratic(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> sq_err = matrix<CPU>::square(probability - expected);

    matrix<CPU> weighted_sq_err = matrix<CPU>::bcast_hadamard_to_stacked_matrix(sq_err, this->weights) * (1 / (float)expected.mat_elements());
    
    return  matrix<CPU>::reduce_sum(weighted_sq_err)[0];
}

matrix<CPU> loss<CPU>::dcross_entropy(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);

    matrix<CPU> prod = expected % matrix<CPU>::reciprocal(probability) * (-1);
    matrix<CPU> weighted_prod = matrix<CPU>::bcast_hadamard_to_stacked_matrix(prod, this->weights);

    return weighted_prod;
}

matrix<CPU> loss<CPU>::dcross_entropy_inkl_softmax(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> err = probability - expected;
    

    matrix<CPU> weighted_err = matrix<CPU>::bcast_hadamard_to_stacked_matrix(err, this->weights);
    

    return weighted_err;
}

matrix<CPU> loss<CPU>::dquadratic(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> err = probability - expected;

    matrix<CPU> weighted_err = matrix<CPU>::bcast_hadamard_to_stacked_matrix(err, this->weights);


    return  weighted_err * 2;
}



loss<CPU>::loss()
{
}

std::function<float(const matrix<CPU>&, const matrix<CPU>& )> loss<CPU>::get_fn(loss_type ltype)
{
    switch (ltype)
    {
        case loss<CPU>::CROSS_ENTROPY:
            return [this](const matrix<CPU>& p, const matrix<CPU>& e) {
                return this->cross_entropy(p, e);
            };

        case loss<CPU>::QUADRATIC:
            return [this](const matrix<CPU>& p, const matrix<CPU>& e) {
                return this->quadratic(p, e);
            };

        default:
            throw std::invalid_argument("Unknown loss type");
    }
}

std::function<matrix<CPU>(const matrix<CPU>&, const matrix<CPU>&)> loss<CPU>::get_derivative_fn(loss_type ltype, activation_type atype)
{
    switch (ltype)
    {
        case loss<CPU>::CROSS_ENTROPY:
        {
            if (atype == activation<CPU>::SOFTMAX)
                return [this](const matrix<CPU>& p, const matrix<CPU>& e) {
                    return this->dcross_entropy_inkl_softmax(p, e);
                };

            else
                return [this](const matrix<CPU>& p, const matrix<CPU>& e) {
                    return this->dcross_entropy(p, e);
                };
        }

        case loss<CPU>::QUADRATIC:
                return [this](const matrix<CPU>& p, const matrix<CPU>& e) {
                    return this->dquadratic(p, e);
                };

        default:
            throw std::invalid_argument("Unknown loss type");
    }
}
