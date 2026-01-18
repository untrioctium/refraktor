set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Quick)

qt_policy(SET QTP0001 NEW)
qt_policy(SET QTP0004 NEW)

qt_standard_project_setup()

qt_add_executable(refrakt-qt src/main.cpp)
set_target_properties(refrakt-qt PROPERTIES AUTOMOC TRUE)

qt_add_qml_module(refrakt-qt URI refrakt-qt-qml VERSION 1.0 QML_FILES qml/main.qml)

target_link_libraries(refrakt-qt PRIVATE Qt6::Quick librefrakt)  