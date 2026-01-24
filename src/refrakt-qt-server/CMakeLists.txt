set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core HttpServer)

qt_standard_project_setup()

qt_add_executable(refrakt-qt-server src/main.cpp)

set_target_properties(refrakt-qt-server PROPERTIES 
    AUTOMOC TRUE
    FIRST_PARTY true
    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_link_libraries(refrakt-qt-server PRIVATE 
    Qt6::Core 
    Qt6::HttpServer
    spdlog
)
