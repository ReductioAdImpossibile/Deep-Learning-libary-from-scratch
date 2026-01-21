#include <iostream>
#include "../headers/tensor.h"
#include <chrono>


float global = 0;
int main()
{
    
    Tensor tens1({10,1}, 0, 1);
    tens1.print();

    std::cout << tens1.max() << std::endl;
    return 0;
}