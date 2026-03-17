# DeepModel - A C++ Deep Learning Libary
> A high performance neural network libary written from scratch in C++, with optional CUDA support.

---
## Overview
This libary is entirely in C++ implemented and has a custom linear algebra engine, which is optimized for CUDA and CPU-only execution.

Every operation like matrix transposing, matrix multiplication and the entire optimization algorithm is implemented by hand.
It also contains a own Dataset class, which gives the user the ability to interpret and edit .csv datasets.

The Github repo contains training examples with mnist and fashion-mnist, while being also benchmarked against pytorch.

---
## Features


### Core Features
- **Backpropagation with L2 regularization**
- **Weighted loss support**
- **Random / Xavier / He weight initalization**
- **ADAM and ADAMW**
- **Dataset editing**


### Optimizers
`ADAM_OPTIMIZER` `STOCHASTIC GRADIENT DESCENT` `BATCH GRADIENT DESCENT` `MINI BATCH GRADIENT DESCENT`

### Activation functions
`RELU`  `IDENTITY`  `ELU`  `SIGMOID`  `LOG_SIGMOID`  `HARD_SIGMOID`  `TANH`  `SOFTMAX`

### Loss functions
`CROSS ENTROPY`  `QUADRATIC (MLE)`


---
## How to build

### Requirements
C++17 GNU / Clang 

OpenMP

CUDA Toolkit (Optional for CUDA version)

Cmake


### CPU-only

```bash
mkdir build
cmake -B build -DENABLE_CUDA=OFF
cmake --build build
```


### CUDA Support

```bash
mkdir build
cmake -B build -DENABLE_CUDA=ON
cmake --build build
```

### Adding your own files

Add this to the 'CMakeLists.txt':
```cmake
add_executable(my_programm my_programm.cpp)
target_link_libaries(my_program PRIVATE DeepModel)
```

Then run: 

```bash
cmake --build build
./build/my_program
```
---

## Quick start

There are mutiple examples to view inside /examples.
Here is the training of a network on the mnist numbers dataset:

**You need to install the mnist dataset as .csv and place it into the dataset folder to run this program.**

link : https://github.com/phoebetronic/mnist

```cpp
#include <iostream>
#include "DeepModel.h"
#include <filesystem>

// Path to the mnist dataset as .csv
const std::string path = "datasets/mnist_train.csv";
 
int main()
{

    if(!std::filesystem::exists(path))
    {
        std::cerr << "Error : Dataset not found at: " << path << " , please edit path or download & place the mnist_train.csv inside /datasets." << std::endl;
        return 1; 
    }

    // Load dataset and edit it
    Dataset data = Dataset(path);
    data.normalize();
    data.one_hot_encode();

    // split the dataset and print information
    auto [train, test] = data.split(0.8);
    test.print_information();


    // Create a new Network
    NeuralNetwork nn;

    nn.configure_input_layer(784);
    nn.add_layer(64, Activation::RELU);
    nn.add_layer(64, Activation::RELU);
    nn.add_layer(10,  Activation::SOFTMAX);
    nn.configure_loss_function(Loss::CROSS_ENTROPY);
    
    // initalise weights
    nn.initalise_he_weights();
    
    // Configure ADAM
    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = 64;

    // Run the Backpropagation
    nn.fit(30, train, adam);

    // Print accuracy
    nn.performance(test);

    nn.save_weights("mnist_example_weights.txt");


}
```

---

## Benchmark

All configurations were run on a mnist Dataset with 60k samples.

#### Architecture:
Neurons                 : 784 x 128 x 128 x 10

Activation functions    : RELU RELU SOFTMAX

Loss function           : Cross entropy

**Screenshots of the results below are inside benchmarks/result_screenshots*
### Mini-Batch SGD 
 
| Batch Size | Epochs | DeepModel (CPU) | DeepModel (CUDA) | PyTorch (CPU) |
|:----------:|:------:|-----------------:|------------------:|--------------:|
| 1  | 1  |  3.49s · 89.6% |  3.92s · 88.2% | 21.26s · 90.9% |
| 2  | 1  |  1.91s · 86.8% |  1.94s · 80.0% | 10.05s · 89.1% |
| 4  | 1  |  1.29s · 83.4% |  1.00s · 39.2% |  5.09s · 83.5% |
| 8  | 1  |  1.14s · 68.2% |  0.51s · 19.7% |  2.59s · 71.9% |
| 16 | 20 | 24.67s · 92.2% |  5.50s · 86.9% | 26.30s · 92.6% |
| 32 | 20 | 36.39s · 88.8% |  3.21s · 77.4% | 13.65s · 90.6% |
| 64 | 20 | 45.38s · 84.8% |  2.40s · 37.2% |  7.44s · 87.4% |


### Mini-Batch ADAM + L2 Regularization

> β₁ = 0.9 · β₂ = 0.999 · ε = 10e-8 · λ = 10e-3

| Batch Size | Epochs | DeepModel (CPU) | DeepModel (CUDA) | PyTorch (CPU) |
|:----------:|:------:|-----------------:|------------------:|--------------:|
| 1  | 1  | 13.32s · 93.9% |  7.56s · 91.6% | 49.61s · 94.5% |
| 2  | 1  |  5.33s · 92.4% |  3.79s · 91.4% | 24.43s · 94.9% |
| 4  | 1  |  2.51s · 94.7% |  1.94s · 88.8% | 11.44s · 94.8% |
| 8  | 1  |  1.95s · 93.5% |  1.01s · 91.1% |  5.62s · 95.4% |
| 16 | 20 | 47.08s · 94.8% | 10.24s · 91.0% | 58.69s · 97.4% |
| 32 | 20 | 53.23s · 96.6% |  5.86s · 92.4% | 32.30s · 97.7% |
| 64 | 20 | 65.42s · 96.5% |  4.26s · 93.3% | 14.92s · 97.5% |

### Interpretation of results


### Run the DeepModel benchmark

```bash
cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark
```

### Run the pytorch benchmark

```bash
python3 benchmark/pytorch_benchmark.py
```
**Requirements** pandas & pytorch






cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build --target mnist_example