cmake_minimum_required(VERSION 3.11...3.18)
enable_language(CXX)
enable_language(C)

include(FetchContent)
include(CheckIPOSupported)
check_ipo_supported()
	
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(CUDAToolkit)

project( 
	refraktor
	VERSION 0.1
	LANGUAGES CXX C CUDA)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.8.5
)
FetchContent_MakeAvailable(spdlog)

add_subdirectory(librefrakt)
add_subdirectory(refraktor)
add_subdirectory(refrakt-server)
add_subdirectory(refrakt-gui)
add_subdirectory(src/refrakt-qt)