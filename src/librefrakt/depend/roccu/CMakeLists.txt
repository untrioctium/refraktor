cmake_minimum_required(VERSION 3.24)

project(roccu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

add_library(roccu OBJECT
  include/roccu.h
  include/roccu_vector_types.h
  include/roccu_cpp_types.h
  src/roccu.cpp)

target_include_directories(roccu PUBLIC include)
set_target_properties(roccu PROPERTIES FIRST_PARTY true)