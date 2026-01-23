#include <iostream>
#include "../headers/tensor.h"
#include "../headers/benchmark.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    //Tensor tens({20000000,1}, 1);

    //std::cout << tens.prod() << std::endl;
    benchmark_tensor_run();
}