set(HEADER_LIST 
	"include/librefrakt/traits/hashable.h"
	"include/librefrakt/traits/noncopyable.h"
	"include/librefrakt/util/hash.h"
	"include/librefrakt/util/cuda.h"
	"include/librefrakt/util/gpuinfo.h"
	"include/librefrakt/util/factory.h"
	"include/librefrakt/util/filesystem.h"
	"include/librefrakt/util/platform.h"
	"include/librefrakt/util/stb.h"
	"include/librefrakt/util/string.h"
	"include/librefrakt/util/zlib.h"
	"include/librefrakt/util/color.h"
	"include/librefrakt/util/nvenc.h"
	"include/librefrakt/util/nvjpeg.h"
	"include/librefrakt/util.h"
	"include/librefrakt/animators.h"
	"include/librefrakt/flame_types.h"
	"include/librefrakt/flame_info.h"
	"include/librefrakt/flame_compiler.h"
	"include/librefrakt/cuda_buffer.h"
	"include/librefrakt/image/tonemapper.h"
	"include/librefrakt/image/denoiser.h"
	"include/librefrakt/image/converter.h"
	"include/ezrtc.h"
	"include/eznve.hpp"
)

add_library(librefrakt STATIC
	"src/util/hash.cpp"
	"src/util/cuda.cpp"
	"src/util/filesystem.cpp"
	"src/util/platform.cpp"
	"src/util/gpuinfo.cpp"
	"src/util/string.cpp"
	"src/util/zlib.cpp"
	"src/util/color.cpp"
	"src/util/stb.cpp"
	"src/util/nvenc.cpp"
	"src/util/nvjpeg.cpp"
	"src/animators.cpp"
	"src/flame_types.cpp"
	"src/flame_info.cpp"
	"src/flame_compiler.cpp"
	"src/image/tonemapper.cpp"
	"src/image/denoiser.cpp"
	"src/image/converter.cpp"
	"src/eznve.cpp"
	${HEADER_LIST}
)

#set_property(TARGET librefrakt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(librefrakt PUBLIC include)
target_include_directories(librefrakt PRIVATE "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0/include")

# yaml-cpp
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_Declare(
  yaml_cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml_cpp)

# xxhash
FetchContent_Declare(
  xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(xxhash)
file(GLOB XXHASH_SOURCES ${xxhash_SOURCE_DIR}/*.c)
list(REMOVE_ITEM XXHASH_SOURCES ${xxhash_SOURCE_DIR}/xxhsum.c)
list(REMOVE_ITEM XXHASH_SOURCES ${xxhash_SOURCE_DIR}/xxh_x86dispatch.c)
add_library(xxhash STATIC ${XXHASH_SOURCES})
target_include_directories(xxhash PUBLIC ${xxhash_SOURCE_DIR})

set(ZLIB_ENABLE_TESTS OFF)
set(ZLIB_COMPAT ON)
# zlib
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng
  GIT_TAG stable
)
FetchContent_MakeAvailable(zlib)
target_include_directories(zlib PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

# sqlite
FetchContent_Declare(
  sqlite3
  GIT_REPOSITORY https://github.com/azadkuh/sqlite-amalgamation
)
FetchContent_MakeAvailable(sqlite3)
target_include_directories(librefrakt PUBLIC ${sqlite3_SOURCE_DIR})

# concurrencpp
FetchContent_Declare(
  concurrencpp
  GIT_REPOSITORY https://github.com/David-Haim/concurrencpp.git
)
FetchContent_MakeAvailable(concurrencpp)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 7.1.2
)
FetchContent_MakeAvailable(fmt)

# stb
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
)
FetchContent_MakeAvailable(stb)
file(GLOB STB_SOURCES ${stb_SOURCE_DIR}/*.c)
add_library(stb STATIC ${STB_SOURCES})
target_include_directories(stb PUBLIC ${stb_SOURCE_DIR})

#cppcodec
FetchContent_Declare(
 cppcodec
 GIT_REPOSITORY https://github.com/tplgy/cppcodec.git
 GIT_TAG master
)
FetchContent_Populate(cppcodec)
target_include_directories(librefrakt PRIVATE ${cppcodec_SOURCE_DIR}/cppcodec)

# inja, note that this also includes nlohmann::json
FetchContent_Declare(
 inja
 GIT_REPOSITORY https://github.com/pantor/inja.git
 GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(inja)

# pugixml
FetchContent_Declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG v1.10
)
FetchContent_MakeAvailable(pugixml)

# ffmpeg's mirror of nvenc headers
FetchContent_Declare(
  nv-codec-headers
  GIT_REPOSITORY https://github.com/FFmpeg/nv-codec-headers.git
)
FetchContent_MakeAvailable(nv-codec-headers)
target_include_directories(librefrakt PRIVATE ${nv-codec-headers_SOURCE_DIR}/include)

FetchContent_Declare(
    dylib
    GIT_REPOSITORY "https://github.com/martin-olivier/dylib"
    GIT_TAG        "v2.1.0"
)
FetchContent_MakeAvailable(dylib)


FetchContent_Declare(
  lua
  GIT_REPOSITORY https://github.com/walterschell/Lua.git
  GIT_TAG master
)
FetchContent_MakeAvailable(lua)

# sol3
FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG v3.2.2
)
FetchContent_MakeAvailable(sol2)

target_compile_definitions(librefrakt PUBLIC EZRTC_USE_ZLIB EZRTC_USE_SQLITE)

target_link_libraries(librefrakt PRIVATE 
	xxhash 
	CUDA::nvrtc
	CUDA::nvjpeg
	stb
	dylib
)

target_link_libraries(librefrakt PUBLIC 
	spdlog 
	CUDA::nvml
	CUDA::cuda_driver
	fmt::fmt
	concurrencpp
	inja
	lua_static
	sol2
	yaml-cpp
	SQLite3
	zlibstatic
	pugixml
)
