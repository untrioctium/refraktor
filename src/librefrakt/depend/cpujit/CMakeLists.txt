cmake_minimum_required(VERSION 3.24)

project(cpujit LANGUAGES CXX)
include(FetchContent)

set(CMAKE_CXX_STANDARD 23)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "cpujit: Found LLVM ${LLVM_PACKAGE_VERSION}")

find_program(LLVM_CLANG
    NAMES clang
    PATHS "${LLVM_TOOLS_BINARY_DIR}"
    NO_DEFAULT_PATH
)
if(NOT LLVM_CLANG)
    message(FATAL_ERROR "Could not find clang in LLVM installation at ${LLVM_TOOLS_BINARY_DIR}")
endif()
message(STATUS "cpujit: Using clang for bitcode: ${LLVM_CLANG}")

llvm_map_components_to_libnames(CPUJIT_LLVM_LIBS
    core
    orcjit
    native
    passes
    support
    transformutils
)

add_library(cpujit SHARED
    src/cpujit.cpp
)

target_include_directories(cpujit PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(cpujit PUBLIC include)
target_compile_definitions(cpujit PRIVATE ${LLVM_DEFINITIONS} CPUJIT_BUILDING)

target_link_libraries(cpujit PRIVATE ${CPUJIT_LLVM_LIBS})

set_target_properties(cpujit PROPERTIES
    OUTPUT_NAME "cpujit"
    FIRST_PARTY true
)

FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    PATCH_COMMAND git restore . && git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sleef-fix-windows-pipes.patch
)

set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_INLINE_HEADERS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sleef)

#set_target_properties(sleef PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(SLEEF_BITCODE_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/sleef_bitcode.c)
set(SLEEF_BITCODE_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sleef_bitcode.ll)
add_custom_command(
    OUTPUT ${SLEEF_BITCODE_OUTPUT}
    COMMAND ${LLVM_CLANG}
        -S -emit-llvm -mavx -mavx2 -mavx512f -mfma
        -O3
        -DSLEEF_INLINE=
        -o ${SLEEF_BITCODE_OUTPUT}
        # Point at the SLEEF build's include dir where sleefinline_*.h lives
        -I${sleef_BINARY_DIR}/include
        ${SLEEF_BITCODE_INPUT}
    DEPENDS ${SLEEF_BITCODE_INPUT} sleef
    COMMENT "Generating SLEEF bitcode for JIT"
    VERBATIM
)
add_custom_target(sleef_bitcode DEPENDS ${SLEEF_BITCODE_OUTPUT})
add_dependencies(cpujit sleef_bitcode)

target_link_directories(cpujit PRIVATE ${CMAKE_INSTALL_PREFIX}/lib)
target_link_libraries(cpujit PRIVATE librefrakt-core)
#target_link_libraries(cpujit PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,sleef>)