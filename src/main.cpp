#include <iostream>
#include "../headers/tensor.h"


int main()
{
    float f = std::numeric_limits<float>::lowest();
    tensor tens1({2,3} , 0 , 1);
    
    tensor tens2({2,3} , 0, 1);

    tensor tens3 = tens1 % tens2;

    tens1.print();
    tens2.print();
    tens3.print();
}