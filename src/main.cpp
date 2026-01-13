#include <iostream>
#include "../headers/tensor.h"


int main()
{
    float f = std::numeric_limits<float>::lowest();
    tensor tens({100000,1} , 1);
    
    std::cout << tens.sum() << std::endl;
    
}