add_executable(refrakt-gui 
  src/refrakt_gui.cpp 
  src/gl.cpp src/gui_styles.cpp 
  include/gl.h 
  include/gui.h 
  include/pch.h 
  include/gui/panels/preview_panel.h 
  src/gui/panels/preview_panel.cpp
  include/gui/modals/render_modal.h
  src/gui/modals/render_modal.cpp)
target_precompile_headers(refrakt-gui PRIVATE include/pch.h)
target_include_directories(refrakt-gui PRIVATE include/)

set_target_properties(refrakt-gui PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# rtmidi
FetchContent_Declare(
  rtmidi
  GIT_REPOSITORY https://github.com/thestk/rtmidi.git
  GIT_TAG master
)
FetchContent_MakeAvailable(rtmidi)


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

#imgui
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)
file(GLOB IMGUI_SOURCES
  ${imgui_SOURCE_DIR}/*.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR})
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw glad)
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

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY    https://github.com/cameron314/readerwriterqueue
  GIT_TAG           master
)
FetchContent_MakeAvailable(readerwriterqueue)
target_link_libraries(refrakt-gui PRIVATE readerwriterqueue)


target_link_libraries(refrakt-gui 
	PRIVATE 
	librefrakt rtmidi Dwmapi.lib)