#include "activationCPU.h"

matrix<CPU> activation<CPU>::identity(const matrix<CPU> &a)
{
    matrix<CPU> result = a;
    return result;
}

matrix<CPU> activation<CPU>::ReLU(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;


    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        where(v < 0, v) = -v;
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a[i];
        result[i] = _d > 0 ? _d : 0;
    }
       

    return result;
}

matrix<CPU> activation<CPU>::ELU(const matrix<CPU> &a)
{


    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;


    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        where(v < 0, v) = ELU_ALPHA_PARAM * (stdx::exp(v) - 1);
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a[i];
        result[i] = _d > 0 ? _d : ELU_ALPHA_PARAM * (std::exp(_d) - 1);
    }

    return result;
}

matrix<CPU> activation<CPU>::Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        v = 1 / (1 + stdx::exp(-v));
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result[i] = 1 / (1 + std::exp(-a[i]));
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Log_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        v = stdx::log(1 / (1 + stdx::exp(-v)));
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result[i] = std::log(1 / (1 + std::exp(-a[i])));
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Hard_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v, _x;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        _x = v;
        v = v / 6 + 1/2;

        where(_x < -3, v) = 0;
        where(_x > 3, v) = 0;

        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
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

matrix<CPU> activation<CPU>::Tanh(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        v = stdx::tanh(v);
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        result[i] = std::tanh(a[i]); 
    }
       

    return result;
}

matrix<CPU> activation<CPU>::Softmax(const matrix<CPU> &a)
{
    
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd exp_sum_simd(0), v;
    matrix<CPU> result = a;
    

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        exp_sum_simd += stdx::exp(-v); 
    }

    float exp_sum = stdx::reduce(exp_sum_simd, std::plus{});
    for(; i < a.size(); i++)
        exp_sum += std::exp(-a[i]);
    
    i = 0;
    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        v = stdx::exp(-v) / exp_sum;
        v.copy_to(&result[i], stdx::element_aligned); 
    }

    for(; i < result.size(); i++)
        result[i] = std::exp(-a[i]) / exp_sum;


    return result;
}

matrix<CPU> activation<CPU>::didentity(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape(), 1);
    return (matrix<CPU>) result;
}

matrix<CPU> activation<CPU>::dReLU(const matrix<CPU> &a)
{
    
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        where(v <= 0, v) = 0;
        where(v > 0, v) = 1;
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a[i];
        
        if(_d <= 0)
            result[i] = 0;
        else
            result[i] = 1;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dELU(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        where(v > 0, v) = 1;
        where(v <= 0, v) = ELU_ALPHA_PARAM * stdx::exp(v);
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
        float _d = a[i];
        if(_d <= 0)
            result[i] = ELU_ALPHA_PARAM * std::exp(_d);
        else
            result[i] = 1;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dSigmoid(const matrix<CPU> &a)
{
    matrix<CPU> sig = Sigmoid(a);
    return sig * (1 - sig);
}

matrix<CPU> activation<CPU>::dLog_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> sig = Sigmoid(a);
    return 1 - sig;
}

matrix<CPU> activation<CPU>::dHard_Sigmoid(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v, _x;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        _x = v;
        where(_x < -3 || _x > 3 , v) = 0;
        where( 3 >=_x && _x >= -3, v) = 1/6;
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {   
        result[i] = a[i] > 3 || a[i] < -3 ? 0 : 1/6;
    }
       
    return result;
}

matrix<CPU> activation<CPU>::dTanh(const matrix<CPU> &a)
{
    matrix<CPU> result(a.get_shape());
    const size_t limit = (a.size() / w) * w;
    size_t i = 0;
    fsimd v;

    for(; i < limit; i += w)
    {
        v.copy_from(&a[i], stdx::element_aligned);
        v = 1 / (stdx::cosh(v) * stdx::cosh(v));
        v.copy_to(&result[i], stdx::element_aligned);
    }
    
    for(; i < a.size(); i++)
    {
    
        result[i] = 1 / (std::cosh(a[i]) * std::cosh(a[i]));
    }
       

    return result;
}






float loss<CPU>::Cross_Entropy(const matrix<CPU> &expected, const matrix<CPU> &probability)
{
    float sum = 0;
    for(int i = 0; i < expected.size(); i++)
        if(expected[i] != 0)
            sum += std::log(probability[i]);
    return sum;
}

float loss<CPU>::Quadratic(const matrix<CPU> &expected, const matrix<CPU> &probability)
{
    float sum = 0;
    const size_t limit = (expected.size() / w) * w;

    size_t i = 0;

    fsimd exp, prob, v, sum_simd(0);
    for(; i < limit; i += w)
    {
        exp.copy_from(&expected[i], stdx::element_aligned );
        prob.copy_from(&probability[i], stdx::element_aligned);

        v = exp - prob;
        v *= v;
        sum_simd += v;
    }

    sum = stdx::reduce(sum_simd, std::plus{});

    for(; i < expected.size(); i++)
        sum += (probability[i] - expected[i]) * (probability[i] - expected[i]);

    return sum * 1/ expected.size();
}

matrix<CPU> loss<CPU>::dCross_Entropy(const matrix<CPU> &expected, const matrix<CPU> &probability)
{
    matrix<CPU> grad(probability.get_shape(), 0);

    #pragma omp parallel for
    for (size_t i = 0; i < probability.size(); ++i)
    {
        if (expected[i] != 0.0f)
            grad[i] = -1 / probability[i];
    }

    return grad;
    
}

matrix<CPU> loss<CPU>::dCross_Entropy_inkl_Softmax(const matrix<CPU> &expected, const matrix<CPU> &probability)
{
    matrix<CPU> grad(probability.get_shape());

    const size_t limit = (expected.size() / w) * w;
    size_t i = 0;
    fsimd prob, exp, v;
    
    for(; i < limit; i++)
    {
        prob.copy_from(&probability[i], stdx::element_aligned);
        exp.copy_from(&expected[i],stdx::element_aligned);

        v = prob - exp;
        v.copy_to(&grad[i], stdx::element_aligned);
    }

    for (; i < expected.size(); i++)
        grad[i] = probability[i] - expected[i];
    return grad;
}

matrix<CPU> loss<CPU>::dQuadratic(const matrix<CPU> &expected, const matrix<CPU> &probability)
{
    matrix<CPU> grad(probability.get_shape());
    const size_t limit = (expected.size() / w) * w;
    size_t i = 0;
    fsimd prob, exp, v;
    fsimd size((float)expected.size());

    for(; i < limit; i++)
    {
        prob.copy_from(&probability[i], stdx::element_aligned);
        exp.copy_from(&expected[i],stdx::element_aligned);

        v = 2 * (prob - exp) / size;
        v.copy_to(&grad[i], stdx::element_aligned);
    }

    for (; i < expected.size(); i++)
        grad[i] = 2 * (probability[i] - expected[i]) / (float)expected.size() ;
    return grad;
}
