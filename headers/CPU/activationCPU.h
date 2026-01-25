#pragma once
#include "activation.h"
#include "matrixCPU.h"



template<>
class activation<CPU>
{   
    static matrix<CPU> ReLU(const matrix<CPU>& a);
    static matrix<CPU> ELU(const matrix<CPU>& a);
    static matrix<CPU> Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Log_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Hard_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Tanh(const matrix<CPU>& a);
    static matrix<CPU> Hard_Tanh(const matrix<CPU>& a);
    static matrix<CPU> Softmax(const matrix<CPU>& a);

    static matrix<CPU> dReLU(const matrix<CPU>& a);
    static matrix<CPU> dELU(const matrix<CPU>& a);
    static matrix<CPU> dSigmoid(const matrix<CPU>& a);
    static matrix<CPU> dLog_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dHard_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dTanh(const matrix<CPU>& a);
    static matrix<CPU> dHard_Tanh(const matrix<CPU>& a);
    static matrix<CPU> dSoftmax(const matrix<CPU>& a);
    
   inline static float ELU_ALPHA_PARAM = 1;
};


