cmake_minimum_required(VERSION 3.16)
project(DEEPLEARNINGLIBARY LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include-Verzeichnis
include_directories(include)

# Executable
add_executable(deeplearninglibary
    src/main.cpp
    src/tensor.cpp
)

# Compile Options (OpenMP + Warnungen)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(deeplearninglibary PRIVATE -fopenmp)
endif()