cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(eznve LANGUAGES CXX)

find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_library(eznve STATIC
  include/eznve.hpp
  src/eznve.cpp
)

target_include_directories(eznve PUBLIC include)

# ffmpeg's mirror of nvenc headers
FetchContent_Declare(
  nv-codec-headers
  GIT_REPOSITORY https://github.com/FFmpeg/nv-codec-headers.git
)
FetchContent_MakeAvailable(nv-codec-headers)
target_include_directories(eznve PRIVATE ${nv-codec-headers_SOURCE_DIR}/include)

target_link_libraries(eznve PUBLIC dylib roccu)