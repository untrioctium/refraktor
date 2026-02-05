cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(ezrtc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_library(ezrtc STATIC
  include/ezrtc.hpp
  src/ezrtc.cpp)

target_include_directories(ezrtc PUBLIC include)

FetchContent_Declare(
  boost_pfr
  GIT_REPOSITORY https://github.com/boostorg/pfr.git
  GIT_TAG develop
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(boost_pfr)
target_link_libraries(ezrtc PUBLIC boost_pfr)

FetchContent_Declare(
  nameof
  GIT_REPOSITORY https://github.com/Neargye/nameof.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nameof)
target_link_libraries(ezrtc PUBLIC nameof)

target_link_libraries(ezrtc PUBLIC roccu)
set_target_properties(ezrtc PROPERTIES FIRST_PARTY true)