cmake_minimum_required(VERSION 3.26.4)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(test LANGUAGES CXX CUDA)

add_library(Vec3 include/Vec3.hpp)
set_target_properties(Vec3 PROPERTIES LINKER_LANGUAGE CXX)

find_package(CUDAToolkit REQUIRED)

add_library(MarchingCubes include/marching_cubes.cu include/marching_cubes.hpp)
set_target_properties(MarchingCubes PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(MarchingCubes PRIVATE cuda cudart)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MarchingCubes)
