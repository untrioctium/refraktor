cmake_minimum_required(VERSION 3.11...3.24)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
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
set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.13.0
  EXCLUDE_FROM_ALL
  GIT_SHALLOW TRUE
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
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(zlib)
#target_include_directories(zlib PUBLIC ${zlib_SOURCE_DIR})
add_library(ZLIB::ZLIB ALIAS zlib)
set(ZLIB_INCLUDE_DIR "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")


add_subdirectory(src/librefrakt)
add_subdirectory(src/refraktor)
add_subdirectory(src/refrakt-server)
add_subdirectory(src/refrakt-gui)

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
	endif()
endforeach()


#install(TARGETS refrakt-gui RUNTIME DESTINATION . COMPONENT refrakt)
#set(CPACK_COMPONENTS_ALL refrakt)
#include(CPack)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()