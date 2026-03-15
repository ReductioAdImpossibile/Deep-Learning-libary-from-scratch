
#include "matrixCUDA.cuh"
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
    matrix<CUDA> result = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height());

    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_relu<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());

    return result;
}

matrix<CUDA> activation<CUDA>::elu(const matrix<CUDA> &a)
{
    matrix<CUDA> result = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_elu<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), ELU_ALPHA_PARAM, result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height());
    
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_sigmoid<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::log_sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height());

    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_log_sigmoid<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::hard_sigmoid(const matrix<CUDA> &a)
{
    matrix<CUDA> result = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height());
   
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_hard_sigmoid<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::tanh(const matrix<CUDA> &a)
{    
    matrix<CUDA> result(a.rows(), a.columns());
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_tanh<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::softmax(const matrix<CUDA> &a)
{
    matrix<CUDA> ones_bcast = matrix<CUDA>::create_stacked_matrix(a.rows(), a.columns(), a.height(), 1);

    matrix<CUDA> max_rows = a.max(); 

    matrix<CUDA> max_expanded = matrix<CUDA>::bcast_scale_to_stacked_matrix(ones_bcast, max_rows);

    matrix<CUDA> exp = matrix<CUDA>::exp(a - max_expanded);

    matrix<CUDA> exp_sum = exp.sum();

    matrix<CUDA> exp_sum_expanded = matrix<CUDA>::bcast_scale_to_stacked_matrix(ones_bcast, exp_sum);

    return exp % matrix<CUDA>::reciprocal(exp_sum_expanded);
}

matrix<CUDA> activation<CUDA>::didentity(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns(), 1);
    return result;
}

matrix<CUDA> activation<CUDA>::drelu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_drelu<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::delu(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_delu<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), ELU_ALPHA_PARAM, result.elements());
    
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
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_dhard_sigmoid<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

matrix<CUDA> activation<CUDA>::dtanh(const matrix<CUDA> &a)
{
    matrix<CUDA> result(a.rows(), a.columns());
    size_t blocks = (result.elements() + matrix<CUDA>::THREADS_1D - 1) / matrix<CUDA>::THREADS_1D;

    activation_function_kernel_dtanh<<<blocks, matrix<CUDA>::THREADS_1D>>>(a.raw(), result.raw(), result.elements());
    
    return result;
}

std::function<matrix<CUDA>(const matrix<CUDA>&)>  activation<CUDA>::get_fn(activation_type atype)
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

std::function<matrix<CUDA>(const matrix<CUDA>&)>  activation<CUDA>::get_derivative_fn(activation_type atype)
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




float loss<CUDA>::cross_entropy(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> prod = matrix<CUDA>::log2(probability) % expected;

    matrix<CUDA> weighted_prod = matrix<CUDA>::bcast_hadamard_to_stacked_matrix(prod, this->weights);

    return matrix<CUDA>::reduce_sum(-1 * weighted_prod.sum())[0];
}

float loss<CUDA>::quadratic(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> sq_err = matrix<CUDA>::square(probability - expected);

    matrix<CUDA> weighted_sq_err = matrix<CUDA>::bcast_hadamard_to_stacked_matrix(sq_err, this->weights) * (1 / (float)expected.mat_elements());
    
    return  matrix<CUDA>::reduce_sum(weighted_sq_err)[0];
}


matrix<CUDA> loss<CUDA>::dcross_entropy(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> grad(probability.rows(), probability.columns(), 0);

    matrix<CUDA> prod = expected % matrix<CUDA>::reciprocal(probability) * (-1);
    matrix<CUDA> weighted_prod = matrix<CUDA>::bcast_hadamard_to_stacked_matrix(prod, this->weights);

    return weighted_prod;
}

matrix<CUDA> loss<CUDA>::dcross_entropy_inkl_softmax(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> err = probability - expected;
    

    matrix<CUDA> weighted_err = matrix<CUDA>::bcast_hadamard_to_stacked_matrix(err, this->weights);
    

    return weighted_err;
}

matrix<CUDA> loss<CUDA>::dquadratic(const matrix<CUDA> &probability, const matrix<CUDA> &expected)
{
    matrix<CUDA> err = probability - expected;

    matrix<CUDA> weighted_err = matrix<CUDA>::bcast_hadamard_to_stacked_matrix(err, this->weights);


    return  weighted_err * 2;
}




loss<CUDA>::loss()
{
}

std::function<float(const matrix<CUDA>&, const matrix<CUDA>& )> loss<CUDA>::get_fn(loss_type ltype)
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

std::function<matrix<CUDA>(const matrix<CUDA>&, const matrix<CUDA>&)> loss<CUDA>::get_derivative_fn(loss_type ltype, activation_type atype)
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
