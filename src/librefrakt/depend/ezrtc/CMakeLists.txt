cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(ezrtc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_library(ezrtc STATIC
  include/ezrtc.h
  src/ezrtc.cpp)

target_include_directories(ezrtc PUBLIC include)

target_link_libraries(ezrtc PUBLIC roccu)
set_target_properties(ezrtc PROPERTIES FIRST_PARTY true)