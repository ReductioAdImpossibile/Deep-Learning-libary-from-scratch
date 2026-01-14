#include <iostream>
#include "../headers/tensor.h"


int main()
{
    float f = std::numeric_limits<float>::lowest();
    tensor tens({2,3,4} , 1);
    
    tensor x  = tens.sum(2);
    x.print();
    
}