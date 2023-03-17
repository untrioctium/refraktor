cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(flang LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_library(flang STATIC
    include/flang/ast_node.h
    include/flang/ast.h
    include/flang/matcher.h
    include/flang/grammar.h
    include/flang.h
    src/grammar_defs.h
    src/ast.cpp
    src/grammar.cpp
    src/matcher.cpp)

FetchContent_Declare(
    pegtl
    GIT_REPOSITORY "https://github.com/taocpp/PEGTL"
    GIT_TAG        "3.2.7"
)
FetchContent_MakeAvailable(pegtl)
target_include_directories(flang PUBLIC include)
target_link_libraries(flang PUBLIC pegtl)