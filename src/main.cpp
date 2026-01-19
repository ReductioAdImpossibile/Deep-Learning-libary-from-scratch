#include <iostream>
#include "../headers/tensor.h"


int main()
{
    
    Tensor tens1({10,10}, 0, 5);
    tens1.print();
    std::cout << tens1.prod() << std::endl;
    
    float sum = 1.0f;
    float* data = tens1.raw();
    size_t n = tens1.get_size();
    for(size_t i=0; i<n; i++) {
        sum *= data[i];
    }
    std::cout << sum << std::endl;

    //tens2.print();
    //Tensor t = tens1 % tens2;


    //t.print();

    return 0;
}