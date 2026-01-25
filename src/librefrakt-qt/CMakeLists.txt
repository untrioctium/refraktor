find_package(Qt6 REQUIRED COMPONENTS Quick)

qt_standard_project_setup()

qt_add_library(librefrakt-qt STATIC
    include/services/kernel_compile_queue.hpp
    include/services/local_render_queue.hpp
    
    src/services/kernel_compile_queue.cpp
    src/services/local_render_queue.cpp
)

qt_add_qml_module(librefrakt-qt
    URI Refrakt.Qt
    VERSION 1.0
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Refrakt/Qt

    SOURCES
        include/services/kernel_compile_queue.hpp
        include/services/local_render_queue.hpp
    )

set_target_properties(librefrakt-qt PROPERTIES
    FIRST_PARTY true
    AUTOMOC TRUE)

target_include_directories(librefrakt-qt PUBLIC 
    include
    include/services)

target_link_libraries(librefrakt-qt PUBLIC
    Qt6::Quick
    librefrakt
    librefrakt-qt-core)