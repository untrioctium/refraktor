project(refrakt-thumbnail-provider)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(refrakt-thumbnail-provider SHARED
    src/dll.cpp
    src/refrakt_thumbnail_provider.cpp
    src/refrakt_thumbnail_provider.h
)

set_target_properties(refrakt-thumbnail-provider PROPERTIES FIRST_PARTY true)

target_link_libraries(refrakt-thumbnail-provider
    PRIVATE
    Shlwapi.lib
    librefrakt
)