set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Quick Concurrent)

qt_policy(SET QTP0001 NEW)
qt_policy(SET QTP0004 NEW)

qt_standard_project_setup()

qt_add_executable(refrakt-qt src/main.cpp)
set_target_properties(refrakt-qt PROPERTIES AUTOMOC TRUE)

qt_add_qml_module(
    refrakt-qt 
    URI Refrakt 
    VERSION 1.0 
    QML_FILES 
        qml/main.qml
    SOURCES
        src/components/flame_preview.cpp
        src/components/flame_preview.hpp

        src/services/animation_database.cpp
        src/services/animation_database.hpp

        src/services/variation_database.cpp
        src/services/variation_database.hpp

        src/services/kernel_compile_queue.cpp
        src/services/kernel_compile_queue.hpp

        src/services/flame_render_queue.cpp
        src/services/flame_render_queue.hpp

        src/models/flame_handle.cpp
        src/models/flame_handle.hpp
    )

target_include_directories(refrakt-qt PRIVATE 
    src 
    src/components 
    src/services
    src/models
)
target_link_libraries(refrakt-qt PRIVATE Qt6::Quick Qt6::Concurrent librefrakt)