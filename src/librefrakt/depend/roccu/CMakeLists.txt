cmake_minimum_required(VERSION 3.24)

project(roccu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

add_library(roccu OBJECT
  include/roccu.hpp
  include/roccu_vector_types.hpp
  include/roccu_cpp_types.hpp
  src/roccu.cpp)

target_include_directories(roccu PUBLIC include)
set_target_properties(roccu PROPERTIES FIRST_PARTY true)