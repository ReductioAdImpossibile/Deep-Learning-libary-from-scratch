#include <iostream>
#include "../headers/tensor.h"


int main()
{
    
    Tensor tens({2,3,40}, 1);
    tens.print();

    std::cout << tens.sum() << std::endl;

    return 0;
}