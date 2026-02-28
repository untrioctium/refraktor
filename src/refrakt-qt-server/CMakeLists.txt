set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core HttpServer WebSockets)

qt_standard_project_setup()

qt_add_executable(refrakt-qt-server
    src/main.cpp
    src/stream_session.hpp
    src/stream_session.cpp
)

set_target_properties(refrakt-qt-server PROPERTIES 
    AUTOMOC TRUE
    FIRST_PARTY true
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)

target_link_libraries(refrakt-qt-server PRIVATE 
    Qt6::Core 
    Qt6::HttpServer
    Qt6::WebSockets
    librefrakt-qt
    librefrakt-impl
    readerwriterqueue
    spdlog
)
