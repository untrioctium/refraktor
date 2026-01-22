add_executable(refrakt-gui 
  src/refrakt_gui.cpp 
  src/gl.cpp src/gui_styles.cpp 
  include/gl.hpp
  include/gui.hpp
  include/pch.hpp
  include/gui/panels/preview_panel.hpp
  src/gui/panels/preview_panel.cpp
  include/gui/modals/render_modal.hpp
  src/gui/modals/render_modal.cpp
  include/gui/panels/timeline.hpp
  src/gui/panels/timeline.cpp
  include/gui/panels/flame_editor.hpp
  src/gui/panels/flame_editor.cpp
  include/command_executor.hpp
  src/command_executor.cpp
)

target_precompile_headers(refrakt-gui PRIVATE include/pch.hpp)
target_include_directories(refrakt-gui PRIVATE include/)

set_target_properties(refrakt-gui PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(refrakt-gui PROPERTIES FIRST_PARTY true)

set(RTMIDI_TARGETNAME_UNINSTALL "rtmidi-uninstall" CACHE STRING "" FORCE)
set(RTMIDI_BUILD_TESTING OFF CACHE BOOL "" FORCE)
# rtmidi
FetchContent_Declare(
  rtmidi
  GIT_REPOSITORY https://github.com/thestk/rtmidi.git
  GIT_TAG 6.0.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(rtmidi)

#implot
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(implot)
file(GLOB IMPLOT_SOURCES ${implot_SOURCE_DIR}/*.cpp)
add_library(implot STATIC ${IMPLOT_SOURCES})
target_include_directories(implot PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot PUBLIC imgui)
target_link_libraries(refrakt-gui PRIVATE implot)

#iconfontcppheaders
FetchContent_Declare(
  iconfontcppheaders
  GIT_REPOSITORY https://github.com/juliettef/IconFontCppHeaders.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(iconfontcppheaders)
target_include_directories(refrakt-gui PUBLIC ${iconfontcppheaders_SOURCE_DIR})

add_subdirectory(depend/imftw)

target_link_libraries(refrakt-gui 
	PRIVATE 
	librefrakt rtmidi imftw reproc++ Dwmapi.lib)