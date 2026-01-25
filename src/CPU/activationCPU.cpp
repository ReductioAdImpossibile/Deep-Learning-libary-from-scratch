
#include "activationCPU.h"


matrix<CPU> activation<CPU>::ReLU(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        where(v < 0, v) = -v;
        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a_data[i];
        result_data[i] = _d > 0 ? _d : 0;
    }
       

    return result;
}

matrix<CPU> activation<CPU>::ELU(const matrix<CPU> &a)
{


        matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        where(v < 0, v) = ELU_ALPHA_PARAM * (stdx::exp(v) - 1);
        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a_data[i];
        result_data[i] = _d > 0 ? _d : ELU_ALPHA_PARAM * (std::exp(_d) - 1);
    }

    return result;
}

matrix<CPU> activation<CPU>::Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        v = 1 / (1 + stdx::exp(-v));
        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result[i] = 1 / (1 + std::exp(-a_data[i]));
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Log_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        v = stdx::log(1 / (1 + stdx::exp(-v)));
        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result_data[i] = std::log(1 / (1 + std::exp(-a_data[i])));
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Hard_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v, _x;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        _x = v;
        v = v / 6 + 1/2;

        where(_x < -3, v) = 0;
        where(_x > 3, v) = 0;

        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a_data[i];
        
        if(_d < -3)
            result[i] = 0;
        else if(_d > 3)
            result[i] = 1;
        else
            result[i] = _d / 6 + 1/2;
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Tanh(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        v = stdx::tanh(v);
        v.copy_to(result_data + i, stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result_data[i] = std::tanh(a_data[i]); 
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Softmax(const matrix<CPU> &a)
{
    matrix<CPU> result;
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd exp_sum_simd(0), v;

    float* a_data = a.raw();
    float* result_data = result.raw();

    for(; i < limit; i += w)
    {
        v.copy_from(a_data + i, stdx::element_aligned);
        exp_sum_simd += stdx::exp(-v); 
    }

    float exp_sum = stdx::reduce(exp_sum_simd, std::plus{});
    for(; i < a.size(); i++)
        exp_sum += std::exp(-a_data[i]);
    

    
    return matrix<CPU>();
}









matrix<CPU> activation<CPU>::dReLU(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dELU(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dSigmoid(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dLog_Sigmoid(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dHard_Sigmoid(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dTanh(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dHard_Tanh(const matrix<CPU> &a)
{
    return matrix<CPU>();
}

matrix<CPU> activation<CPU>::dSoftmax(const matrix<CPU> &a)
{
    return matrix<CPU>();
}
