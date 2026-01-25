#include <iostream>
#include "modelCPU.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    Matrix mat({2,3}, 0, 1);

    Matrix mat2 = mat;
    mat[0] = 1;

    mat.print();
    mat2.print();
}