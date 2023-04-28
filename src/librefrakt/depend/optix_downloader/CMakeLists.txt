cmake_minimum_required(VERSION 3.24)

find_package(PythonInterp REQUIRED)
project(optix LANGUAGES CXX)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "Downloading Optix headers to ${CMAKE_CURRENT_BINARY_DIR}/include")

set(OPTIX_HEADERS 
 "optix.h" 
 "optix_stubs.h" 
 "optix_denoiser_tiling.h" 
 "optix_device.h" 
 "optix_function_table.h" 
 "optix_function_table_definition.h"
 "optix_host.h"
 "optix_stack_size.h"
 "optix_types.h")

foreach(OPTIX_HEADER IN LISTS OPTIX_HEADERS)
    message(STATUS "Downloading Optix header: ${OPTIX_HEADER}")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/download_optix_header.py" ${OPTIX_HEADER}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include
        RESULT_VARIABLE OPTIX_DOWNLOAD_RESULT
        OUTPUT_VARIABLE OPTIX_DOWNLOADED_HEADERS)

endforeach()


set(ENV{OPTIX_INCLUDE_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/include)