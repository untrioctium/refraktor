find_package(Qt6 REQUIRED COMPONENTS Quick Concurrent)

qt_standard_project_setup()

qt_add_library(librefrakt-qt STATIC
    include/services/kernel_compile_queue.hpp
    include/services/local_render_queue.hpp
    include/services/local_flame_directory.hpp

    src/services/kernel_compile_queue.cpp
    src/services/local_render_queue.cpp
    src/services/local_flame_directory.cpp
)


qt_add_qml_module(librefrakt-qt
    URI Refrakt.Qt
    VERSION 1.0
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Refrakt/Qt

    SOURCES
        include/services/kernel_compile_queue.hpp
        include/services/local_render_queue.hpp
        include/services/local_flame_directory.hpp
    )

set_target_properties(librefrakt-qt PROPERTIES
    FIRST_PARTY true
    AUTOMOC TRUE)

target_include_directories(librefrakt-qt PUBLIC 
    include
    include/services)

target_link_libraries(librefrakt-qt PUBLIC
    Qt6::Quick
    Qt6::Concurrent
    librefrakt
    librefrakt-qt-core)