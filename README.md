# DeepModel - A C++ Deep Learning Libary
> A high performance neural network libary written from scratch in C++, with optional CUDA support.

---
## Overview
This libary is entirely in C++ implemented and has a custom linear algebra engine, which is optimized for CUDA and CPU-only execution.\n

Every operation like matrix transposing, matrix multiplication and the entire optimization algorithm is implemented by hand.

The Github repo contains training examples with mnist and fashion-mnist, while being also benchmarked against pytorch.

---
## Features


### Core Features
- **Backpropagation with L2 regularization**
- **Weighted loss support**
- **Random / Xavier / He weight initalization**
- Own **Dataset** class, which allows operations like one hot encoding, normalization, spliting


### Optimizers
`ADAM_OPTIMIZER`, `STOCHASTIC GRADIENT DESCENT`, `BATCH GRADIENT DESCENT`, `MINI BATCH GRADIENT DESCENT`

### Activation functions
`RELU`, `IDENTITY`, `ELU`, `SIGMOID`, `LOG_SIGMOID`, `HARD_SIGMOID`, 
`TANH`, `SOFTMAX`

### Loss function
`CROSS ENTROPY`, `QUADRATIC (MLE)`




cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark

cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build --target mnist_example