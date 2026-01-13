cmake_minimum_required(VERSION 3.16)
project(DEEPLEARNINGLIBARY LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

add_executable(
    test1
    src/main.cpp
    src/tensor.cpp
    )

