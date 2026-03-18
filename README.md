# DeepModel - A C++ Deep Learning Libary
> A high performance neural network libary written from scratch in C++, with optional CUDA support.
It implements a static Backpropagation algorithm to train feed forward neuralnetworks with simple topology.

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

All configurations were run on a mnist Dataset with 60k samples and the same hyperparameters.
**Furthermore, all random number calculations were performed on the same seed to ensure the replicability of the results.**
You can also run them yourself. Everything except the mnist dataset is included in /benchmarks.


#### Network Architecture:
Neurons                 : 784 x 128 x 128 x 10

Activation functions    : RELU RELU SOFTMAX

Loss function           : Cross entropy



### Mini-Batch SGD
 
| Batch Size | Epochs | DeepModel (CPU) | DeepModel (CUDA) | PyTorch (CPU) |
|:----------:|:------:|-----------------:|------------------:|--------------:|
| 1  | 1  |  3.19s · 89.3% |  3.70s · 86.3% | 26.83s · 91.4% |
| 2  | 1  |  1.81s · 86.8% |  1.87s · 78.1% | 11.44s · 89.1% |
| 4  | 1  |  1.28s · 83.1% |  1.00s · 42.3% |  5.77s · 86.1% |
| 8  | 1  |  1.16s · 74.9% |  0.53s · 18.2% |  2.95s · 69.8% |
| 16 | 20 | 26.05s · 91.9% |  5.50s · 86.0% | 30.29s · 92.5% |
| 32 | 20 | 34.81s · 89.0% |  3.19s · 75.2% | 15.88s · 90.1% |
| 64 | 20 | 46.67s · 85.2% |  2.41s · 49.3% |  8.88s · 86.9% |
 
### Mini-Batch ADAM + L2 Regularization
 
> β₁ = 0.9 · β₂ = 0.999 · ε = 10e-8 · λ = 10e-4
 
| Batch Size | Epochs | DeepModel (CPU) | DeepModel (CUDA) | PyTorch (CPU) |
|:----------:|:------:|-----------------:|------------------:|--------------:|
| 1  | 1  | 12.61s · 92.9% |  7.30s · 92.3% | 68.96s · 95.0% |
| 2  | 1  |  5.04s · 94.5% |  3.75s · 91.2% | 30.95s · 95.7% |
| 4  | 1  |  2.59s · 93.7% |  1.95s · 92.7% | 15.94s · 95.1% |
| 8  | 1  |  2.24s · 93.1% |  0.97s · 89.2% |  7.05s · 95.6% |
| 16 | 20 | 49.16s · 96.5% |  9.97s · 92.8% | 86.18s · 98.0% |
| 32 | 20 | 57.35s · 97.0% |  5.59s · 93.0% | 42.02s · 97.7% |
| 64 | 20 | 64.51s · 96.4% |  4.13s · 93.0% | 19.40s · 97.7% |

### Interpretation of results

DeepModel outperforms PyTorch (CPU) in training speed on both CPU and CUDA with speedups of 2 - 10, but achieves lower accuracy on most configurations.
The primary reason for the longer runtimes of PyTorch and the accuracy gap, is the autograd system of PyTorch.
During the forward pass, PyTorch constructs a dynamic direct graph, which contains every mathmatical operations and its dependencies.
When the backpropagation begins, the gradients are computed by traversing this graph backwards with the chain rule.
This system is way more flexible and is generalised for any other network architecture (e. g: CNN's etc.).

An additional factor is that the Pytorch training loops runs in Python, which is a interpreted language.
DeepModel is written in compiled C++, which removes this kind of overhead completely.

**DeepModel is a static backpropagation algorithm, which can only run a simple feed forward topology and is not as flexible as PyTorch.**
Due to the missing overhead and small batch sizes, DeepModel outperformes PyTorch in this case.

Source : https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

```python3

```


### Run the benchmark for yourself

If u wish to run the benchmark.

#### DeepModel : CPU-only

```bash
cmake -B build -DENABLE_CUDA=OFF -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark/deepmodel_benchmark
```

#### DeepModel : CUDA Support

```bash
cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark/deepmodel_benchmark
```


#### Pytorch : CPU-only 

```bash
python3 benchmark/pytorch_benchmark.py
```
**Requirements** pandas & pytorch

---


## Examples

There are 3 examples.

1. Training on the mnist dataset.
2. Training on the fashion-mnist dataset.
3. Matrix operations with the linear algebra engine.

### Build

#### CPU-only

```bash
cmake -B build -DENABLE_CUDA=OFF -DBUILD_EXAMPLES=ON
cmake --build build
```

#### CUDA Support

```bash
cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build
```

### run:

#### 1 mnist:
```bash
./build/mnist
```
**Requires the mnist dataset as .csv**
Place it into /datasets with name 'mnist_train.csv'.

#### 2 fasion-mnist:
```bash
./build/fashion_nist
```
**Requires the fashion mnist dataset as .csv**
Place it into /datasets with name 'fasion_mnist.csv'.

##### 3 linear algebra:
```bash
./build/linear_algebra
```






