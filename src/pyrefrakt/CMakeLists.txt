include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.1  # or latest stable
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_pyrefrakt src/bindings.cpp)

target_link_libraries(_pyrefrakt PRIVATE librefrakt librefrakt-impl)

set_target_properties(_pyrefrakt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyrefrakt"
)