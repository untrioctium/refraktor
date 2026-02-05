cmake_minimum_required(VERSION 3.11...3.24)
project(refrakt)


cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
enable_language(CXX)
enable_language(C)

#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#set(CMAKE_INSTALL_PREFIX "${CMAKE_HOME_DIRECTORY}/dist")
#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

include(FetchContent)
include(CheckIPOSupported)
check_ipo_supported()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    link_libraries(stdc++exp)  # Global, or use target_link_libraries per target
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,cppcoreguidelines-*)

#find_package(CUDAToolkit)

# spdlog
set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.13.0
  EXCLUDE_FROM_ALL
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

# zlib-ng as zlib replacement
set(ZLIB_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ZLIB_COMPAT ON CACHE BOOL "" FORCE)
set(ZLIB_ALIASES ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  zlib-ng
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng
  GIT_TAG stable
  OVERRIDE_FIND_PACKAGE
  EXCLUDE_FROM_ALL
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(zlib-ng)

add_subdirectory(src/librefrakt-core)
add_subdirectory(src/librefrakt)
add_subdirectory(src/librefrakt-qt-core)
add_subdirectory(src/librefrakt-qt)
#add_subdirectory(src/librefrakt-qt-server)
add_subdirectory(src/refrakt-qt)
add_subdirectory(src/refrakt-qt-server)
add_subdirectory(src/pyrefrakt)
#add_subdirectory(src/refrakt-gui)

function(get_all_targets var)
    set(targets)
    get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

get_all_targets(all_targets) 

foreach(target_name ${all_targets})
	get_target_property(is_first_party ${target_name} FIRST_PARTY)
	if(NOT ${is_first_party} STREQUAL true)
		set_property(TARGET ${target_name} PROPERTY FOLDER "Dependencies")
	else()
		set_target_properties(${target_name} PROPERTIES
    CXX_CLANG_TIDY "clang-tidy"
		)
	endif()
endforeach()


#install(TARGETS refrakt-gui RUNTIME DESTINATION . COMPONENT refrakt)
#set(CPACK_COMPONENTS_ALL refrakt)
#include(CPack)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()