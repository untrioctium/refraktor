cmake_minimum_required(VERSION 3.11...3.18)
enable_language(CXX)
enable_language(C)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_INSTALL_PREFIX "${CMAKE_HOME_DIRECTORY}/dist")
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

include(FetchContent)
include(CheckIPOSupported)
check_ipo_supported()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

project(refrakt)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#find_package(CUDAToolkit)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.8.5
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(spdlog)

set(ZLIB_ENABLE_TESTS OFF)
set(ZLIB_COMPAT ON)
# zlib
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng
  GIT_TAG stable
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(zlib)
target_include_directories(zlib PUBLIC ${zlib_SOURCE_DIR})
add_library(ZLIB::ZLIB ALIAS zlib)
set(ZLIB_INCLUDE_DIR "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")

add_subdirectory(src/librefrakt)
add_subdirectory(src/refraktor)
add_subdirectory(src/refrakt-server)
add_subdirectory(src/refrakt-gui)


#install(TARGETS refrakt-gui RUNTIME DESTINATION . COMPONENT refrakt)
#set(CPACK_COMPONENTS_ALL refrakt)
#include(CPack)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()