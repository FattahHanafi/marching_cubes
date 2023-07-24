cmake_minimum_required(VERSION 3.26.4)
set(CMAKE_BUILD_TYPE Release)
set(PROPERTY CXX_STANDARD 23)

project(MarchingCubes)

add_library(${PROJECT_NAME} STATIC marching_cubes.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER marching_cubes.hpp LINKER_LANGUAGE CXX)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
