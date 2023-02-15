add_executable(refrakt-gui src/refrakt_gui.cpp src/gl.cpp src/gui_styles.cpp include/gl.h include/gui.h include/pch.h)
target_precompile_headers(refrakt-gui PRIVATE include/pch.h)
target_include_directories(refrakt-gui PRIVATE include/)

set_target_properties(refrakt-gui PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

FetchContent_Declare(
  eznve
  GIT_REPOSITORY https://github.com/untrioctium/eznve.git
  GIT_TAG main
)
FetchContent_MakeAvailable(eznve)

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
target_link_libraries(refrakt-gui PRIVATE glfw)

# glad
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG v0.1.34
)
FetchContent_MakeAvailable(glad)
target_link_libraries(refrakt-gui PRIVATE glad)

#imgui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)
file(GLOB IMGUI_SOURCES
  ${imgui_SOURCE_DIR}/*.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw)
target_link_libraries(imgui PUBLIC glad)
target_link_libraries(refrakt-gui PRIVATE imgui)

#implot
FetchContent_Declare(
 implot
 GIT_REPOSITORY https://github.com/epezent/implot.git
 GIT_TAG master
)
FetchContent_MakeAvailable(implot)
file(GLOB IMPLOT_SOURCES ${implot_SOURCE_DIR}/*.cpp)
add_library(implot STATIC ${IMPLOT_SOURCES})
target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)
target_link_libraries(refrakt-gui PRIVATE implot)

target_link_libraries(refrakt-gui 
	PRIVATE 
	librefrakt 
	fmt::fmt
      eznve)