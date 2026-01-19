cmake_minimum_required(VERSION 3.16) 
project(DEEPLEARNINGLIBARY LANGUAGES CXX) 
# C++ Standard 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 

# Include-Verzeichnis 
include_directories(include) # Executable 
add_executable( deeplearninglibary src/main.cpp src/tensorCPU.cpp ) 

# Compile Options (OpenMP + Warnungen) 
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deeplearninglibary PRIVATE OpenMP::OpenMP_CXX)
endif()