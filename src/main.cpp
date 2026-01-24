#include <iostream>
#include "tensor.h"
#include "matrix.h"
#include "benchmark.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    Matrix mat1({2,1}, 0, 1);
    Matrix mat2({2,4}, 0,1);
    Matrix mat3({4,4});


    mat1.print();
    std::cout <<  mat1.prod() << std::endl;
    //mat2.print();

    //Matrix::mat_mul(mat1, mat2, mat3);
    //std::cout << std::endl;
    //mat3.print();
}