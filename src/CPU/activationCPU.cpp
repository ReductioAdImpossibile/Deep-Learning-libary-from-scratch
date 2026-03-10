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
    matrix<CPU> result(a.rows(), a.columns());
    
    for(int i = 0;i < a.size(); i++)
        result[i] = a[i] > 0 ? a[i] : 0; 
       
    return result;
}

matrix<CPU> activation<CPU>::elu(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());
    
    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] > 0 ? a[i] : ELU_ALPHA_PARAM * (std::exp(a[i]) - 1);

    return result;
}

matrix<CPU> activation<CPU>::sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());
    
    
    for(int i = 0; i < a.size(); i++)
        result[i] = 1 / (1 + std::exp(-a[i]));
    
    return result;
}

matrix<CPU> activation<CPU>::log_sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());
    
    for(int i = 0; i < a.size(); i++)
        result[i] = std::log(1 / (1 + std::exp(-a[i])));
    
    return result;
}

matrix<CPU> activation<CPU>::hard_sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());
    
    for(int i = 0; i < a.size(); i++)
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
    matrix<CPU> result(a.rows(), a.columns());
    for(int i = 0; i < a.size(); i++)
    {
        result[i] = std::tanh(a[i]); 
    }
       

    return result;
}

matrix<CPU> activation<CPU>::softmax(const matrix<CPU> &a)
{
    matrix<CPU> result = a;

    // 1. Finde den Maximalwert in der Matrix
    float max_val = a[0];
    for(int i = 1; i < a.size(); i++) {
        if(a[i] > max_val) max_val = a[i];
    }

    // 2. Summe berechnen mit (a[i] - max_val)
    float exp_sum = 0;
    for(int i = 0; i < a.size(); i++) {
        exp_sum += std::exp(a[i] - max_val);
    }

    // 3. Ergebnis berechnen
    for(int i = 0; i < result.size(); i++) {
        result[i] = std::exp(a[i] - max_val) / exp_sum;
    }

    return result;
}

matrix<CPU> activation<CPU>::didentity(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns(), 1);
    return (matrix<CPU>) result;
}

matrix<CPU> activation<CPU>::drelu(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());

    for(int i = 0; i < a.size(); i++)
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
    matrix<CPU> result(a.rows(), a.columns());
    
    for(int i = 0; i < a.size(); i++)
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

    matrix<CPU> result(a.rows(), a.columns());
    for(int i = 0; i < a.size(); i++)
    {   
        result[i] = a[i] > 3 || a[i] < -3 ? 0 : 1/6;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dtanh(const matrix<CPU> &a)
{
    matrix<CPU> result(a.rows(), a.columns());
    for(int i = 0; i < a.size(); i++)
    {
        result[i] = 1 / (std::cosh(a[i]) * std::cosh(a[i]));
    }
       

    return result;
}

activation_fn activation<CPU>::get_fn(activation_type atype)
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

activation_fn activation<CPU>::get_derivative_fn(activation_type atype)
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
    float sum = 0;
    float epsilon = 1e-15f; 

    for(int i = 0; i < expected.size(); i++)
    {
        if(expected[i] > 0)
        {
            float p = std::max(probability[i], epsilon);
            sum -= this->weights[i] * expected[i] * std::log(p);
        }
    }
    return sum;
}

float loss<CPU>::quadratic(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    float sum = 0;
    matrix<CPU> err_sq = matrix<CPU>::square(probability - expected);

    return  (err_sq % this->weights).sum() / expected.size();
}


matrix<CPU> loss<CPU>::dcross_entropy(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < probability.size(); ++i)
    {
        if (expected[i] != 0.0f)
            grad[i] = -this->weights[i] / probability[i];
    }

    return grad;
}

matrix<CPU> loss<CPU>::dcross_entropy_inkl_softmax(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    /*
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);
    for (int i = 0; i < expected.size(); i++)
        grad[i] = probability[i] - expected[i];
    */
    
    
    matrix<CPU> grad = (probability - expected) % this->weights;
    return grad;
}

matrix<CPU> loss<CPU>::dquadratic(const matrix<CPU> &probability, const matrix<CPU> &expected)
{
    /*
    matrix<CPU> grad(probability.rows(), probability.columns(), 0);

    for (int i = 0; i < expected.size(); i++)
        grad[i] = 2 * (probability[i] - expected[i]);
    */
    matrix<CPU> grad = 2 * ((probability - expected) % this->weights); 
    return grad;
}




loss<CPU>::loss()
{
}

loss_fn loss<CPU>::get_fn(loss_type ltype)
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

loss_derivative_fn loss<CPU>::get_derivative_fn(loss_type ltype, activation_type atype)
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
