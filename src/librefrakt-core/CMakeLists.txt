project(librefrakt-core LANGUAGES CXX)

set(HEADER_LIST 
    "include/librefrakt/traits/hashable.hpp"
    "include/librefrakt/traits/noncopyable.hpp"
    "include/librefrakt/util/hash.hpp"
    "include/librefrakt/util/zlib.hpp"
    "include/librefrakt/util/color.hpp"
    "include/librefrakt/util/stb.hpp"
    "include/librefrakt/util/factory.hpp"
    "include/librefrakt/anima.hpp"
    "include/librefrakt/flame_types.hpp"
    "include/librefrakt/flame_info.hpp"
    "include/librefrakt/factory.hpp"
    "include/librefrakt/util.hpp"
    "include/librefrakt/vector_types.hpp"
)

add_library(librefrakt-core STATIC
    "src/util/color.cpp"
    "src/util/hash.cpp"
    "src/util/stb.cpp"
    "src/util/zlib.cpp"
    "src/anima.cpp"
    "src/flame_info.cpp"
    "src/flame_types.cpp"
    ${HEADER_LIST}
)

target_include_directories(librefrakt-core PUBLIC include)
set_target_properties(librefrakt-core PROPERTIES FIRST_PARTY true)

# yaml-cpp
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_Declare(
  yaml_cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(yaml_cpp)

# xxhash
FetchContent_Declare(
  xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG v0.8.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(xxhash)
file(GLOB XXHASH_SOURCES ${xxhash_SOURCE_DIR}/*.c)
list(REMOVE_ITEM XXHASH_SOURCES ${xxhash_SOURCE_DIR}/xxhsum.c)
list(REMOVE_ITEM XXHASH_SOURCES ${xxhash_SOURCE_DIR}/xxh_x86dispatch.c)
add_library(xxhash STATIC ${XXHASH_SOURCES})
target_include_directories(xxhash PUBLIC ${xxhash_SOURCE_DIR})

# sqlite
set(ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_SHELL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sqlite3
  GIT_REPOSITORY https://github.com/rhuijben/sqlite-amalgamation
  GIT_TAG 3.45.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(sqlite3)
target_include_directories(librefrakt-core PUBLIC ${sqlite3_SOURCE_DIR})

# glm
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 1.0.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)

# stb
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(stb)
file(GLOB STB_SOURCES ${stb_SOURCE_DIR}/*.c)
add_library(stb STATIC ${STB_SOURCES})
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})

#cppcodec
FetchContent_Declare(
 cppcodec
 GIT_REPOSITORY https://github.com/tplgy/cppcodec.git
 GIT_TAG 8019b8b580f8573c33c50372baec7039dfe5a8ce
 GIT_SHALLOW TRUE
)
FetchContent_Populate(cppcodec)
target_include_directories(librefrakt-core PRIVATE ${cppcodec_SOURCE_DIR}/cppcodec)

# inja, note that this also includes nlohmann::json
FetchContent_Declare(
 inja
 GIT_REPOSITORY https://github.com/pantor/inja.git
 GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(inja)

# pugixml
FetchContent_Declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG v1.14
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pugixml)

set(LUA_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(LUA_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(LUA_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  lua
  GIT_REPOSITORY https://github.com/walterschell/Lua.git
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(lua)

# sol3
FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)

# dlib
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v20.0
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY  YES   # Just fetch, don't add_subdirectory()
)

FetchContent_Populate(dlib)
target_include_directories(librefrakt-core PRIVATE ${dlib_SOURCE_DIR})

FetchContent_MakeAvailable(sol2)

add_subdirectory(depend/flang)

target_link_libraries(librefrakt-core PRIVATE
    xxhash
    stb
    pugixml
    yaml-cpp
)

target_link_libraries(librefrakt-core PUBLIC
    flang
    fmt::fmt
    spdlog
    lua_static
    sol2
    zlib
    glm
    sqlite3-static
    inja
)