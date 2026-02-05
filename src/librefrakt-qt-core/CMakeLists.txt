find_package(Qt6 REQUIRED COMPONENTS Quick Concurrent)

qt_standard_project_setup()

qt_add_library(librefrakt-qt-core STATIC
    include/models/flame_handle.hpp
    include/services/animation_database.hpp
    include/services/variation_database.hpp
    include/services/flame_render_service.hpp
    include/services/flame_directory_service.hpp
    include/services/remote_flame_directory.hpp
    include/services/service_locator.hpp

    src/models/flame_handle.cpp
    src/services/animation_database.cpp
    src/services/variation_database.cpp
    src/services/remote_flame_directory.cpp

    include/services/remote_render_service.hpp
    src/services/remote_render_service.cpp
)

qt_add_qml_module(librefrakt-qt-core
    URI Refrakt.Core
    VERSION 1.0
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Refrakt/Core

    SOURCES
        include/models/flame_handle.hpp
        include/services/animation_database.hpp
        include/services/variation_database.hpp
        include/services/flame_render_service.hpp
        include/services/remote_render_service.hpp
        include/services/flame_directory_service.hpp
        include/services/remote_flame_directory.hpp
    )

set_target_properties(librefrakt-qt-core PROPERTIES
    FIRST_PARTY true
    AUTOMOC TRUE)

target_include_directories(librefrakt-qt-core PUBLIC 
    include
    include/models
    include/services)

target_link_libraries(librefrakt-qt-core PUBLIC
    Qt6::Quick
    Qt6::Concurrent
    librefrakt-core)