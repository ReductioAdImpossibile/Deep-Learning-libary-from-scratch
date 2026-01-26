#include <iostream>
#include "modelCPU.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    Matrix mat1({4,4}, 1);
    Matrix mat2({4,4}, 1);

    Matrix mat3 = mat1 * mat2;
    mat3.print();
}