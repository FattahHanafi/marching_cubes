cmake_minimum_required(VERSION 3.26.4)
set(CMAKE_BUILD_TYPE Release)
set(CXX_STANDARD 20)

project(test)

add_executable(test test.cpp)
target_link_libraries(test Vec3 MarchingCubes)
