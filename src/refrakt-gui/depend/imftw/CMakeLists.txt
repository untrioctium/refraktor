cmake_minimum_required(VERSION 3.24)
include(FetchContent)

project(imftw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

find_package(ZLIB)

add_library(imftw STATIC
    include/imftw/gui.h
    include/imftw/imftw.h
    include/imftw/signals.h
    include/imftw/kvstore.h
    src/events.cpp
    src/events.h
    src/imftw_internal.h
    src/imftw.cpp
    src/signals_internal.h
    src/signals.cpp
    src/kvstore.cpp)

target_include_directories(imftw PUBLIC include)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# glad
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG master
)
FetchContent_MakeAvailable(glad)

# freetype
FetchContent_Declare(
  freetype
  GIT_REPOSITORY https://gitlab.freedesktop.org/freetype/freetype.git
  GIT_TAG VER-2-13-0
)
FetchContent_MakeAvailable(freetype)
target_include_directories(freetype PRIVATE ${ZLIB_INCLUDE_DIR})
target_link_libraries(freetype PUBLIC ${ZLIB_LIBRARIES})

#imgui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)
file(GLOB IMGUI_SOURCES
  ${imgui_SOURCE_DIR}/*.cpp
  ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR}/backends)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR}/misc/freetype)
target_compile_definitions(imgui PUBLIC IMGUI_ENABLE_FREETYPE)
target_compile_definitions(imgui PUBLIC IMGUI_USE_WCHAR32)
target_link_libraries(imgui PRIVATE glfw glad)
target_link_libraries(imgui PUBLIC freetype)

# concurrentqueue
FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG master
)
FetchContent_MakeAvailable(concurrentqueue)

# readerwriterqueue
FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG master
)
FetchContent_MakeAvailable(readerwriterqueue)

target_link_libraries(imftw PRIVATE glfw)
target_link_libraries(imftw PUBLIC glad imgui concurrentqueue readerwriterqueue)