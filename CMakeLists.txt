cmake_minimum_required(VERSION 3.16) 
project(DEEPLEARNINGLIBARY LANGUAGES CXX) 
# C++ Standard 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 

#
include_directories(headers)
include_directories(headers/CPU)
include_directories(headers/CUDA)

add_executable( deeplearninglibary 
    src/main.cpp 
    src/CPU/matrixCPU.cpp 
    src/CPU/activationCPU.cpp
    src/CPU/classificatorCPU.cpp
    src/CPU/datasetCPU.cpp
    ) 

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deeplearninglibary PRIVATE OpenMP::OpenMP_CXX)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(deeplearninglibary PRIVATE
        -Ofast              
        -march=native
        -mtune=native
        -funroll-loops
        -fomit-frame-pointer
        -flto
        -fno-signed-zeros
        -fno-trapping-math
        -fassociative-math
    )
endif()