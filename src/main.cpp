#include <iostream>
#include "modelCPU.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    classificator<CPU> c;
    c.configure_input_layer(12);
    
}