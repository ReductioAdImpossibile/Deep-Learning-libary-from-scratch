cmake_minimum_required(VERSION 3.16) 
project(DEEPLEARNINGLIBARY LANGUAGES CXX) 
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 


option(ENABLE_CUDA "CUDA Support" OFF)

include_directories(headers)


if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    include_directories(headers/CUDA)
    add_compile_definitions(ENABLE_CUDA)

    set(SOURCES 
        src/main.cpp
        src/CUDA/kernel.cu
        src/CUDA/datasetCUDA.cu
        src/CUDA/matrixCUDA.cu
        src/CUDA/activationCUDA.cu
        src/CUDA/neuralnetworkCUDA.cu
    )
else()
    include_directories(headers/CPU)
    set(SOURCES 
        src/main.cpp 
        src/CPU/dataset.cpp
        src/CPU/matrixCPU.cpp 
        src/CPU/activationCPU.cpp
        src/CPU/neuralnetworkCPU.cpp
    )
endif()


add_executable(DEEPLEARNINGLIBARY ${SOURCES})

if (NOT ENABLE_CUDA AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(DEEPLEARNINGLIBARY PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_compile_options(DEEPLEARNINGLIBARY PRIVATE
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


if(CUDA_ENABLED)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(DEEPLEARNINGLIBARY PRIVATE 
        CUDA::cudart
    )

endif()