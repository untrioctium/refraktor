cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(roccu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_library(roccu STATIC
  include/roccu.h
  include/roccu_vector_types.h
  include/roccu_cpp_types.h
  src/roccu.cpp)

target_include_directories(roccu PUBLIC include)
target_link_libraries(roccu PUBLIC dylib)
set_target_properties(roccu PROPERTIES FIRST_PARTY true)