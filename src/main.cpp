#include <iostream>
#include "../headers/tensor.h"


int main()
{
    tensor tens1({2,3} , -1, 1);   
    tensor tens2({2,3} , 1);

    tens1.print();
    std::cout << tens1.L1() << std::endl;
}