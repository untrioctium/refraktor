set(HEADER_LIST 
	"include/librefrakt/traits/hashable.h"
	"include/librefrakt/traits/noncopyable.h"
	"include/librefrakt/util/hash.h"
	"include/librefrakt/util/cuda.h"
	"include/librefrakt/util/factory.h"
	"include/librefrakt/util/filesystem.h"
	"include/librefrakt/util/platform.h"
	"include/librefrakt/util/stb.h"
	"include/librefrakt/util/string.h"
	"include/librefrakt/util/zlib.h"
	"include/librefrakt/util/color.h"
	"include/librefrakt/util/nvenc.h"
	"include/librefrakt/util.h"
	"include/librefrakt/animators.h"
	"include/librefrakt/kernel_manager.h"
	"include/librefrakt/flame_types.h"
	"include/librefrakt/flame_info.h"
	"include/librefrakt/flame_compiler.h"
	"include/librefrakt/cuda_buffer.h"
)

add_library(librefrakt STATIC
	"src/util/hash.cpp"
	"src/util/cuda.cpp"
	"src/util/filesystem.cpp"
	"src/util/platform.cpp"
	"src/util/string.cpp"
	"src/util/zlib.cpp"
	"src/util/color.cpp"
	"src/util/stb.cpp"
	"src/util/nvenc.cpp"
	"src/kernel_manager.cpp"
	"src/animators.cpp"
	"src/flame_types.cpp"
	"src/flame_info.cpp"
	"src/flame_compiler.cpp"
	${HEADER_LIST}
)

set_property(TARGET librefrakt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(librefrakt PUBLIC include)

# yaml-cpp
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
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

# zlib
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng
  GIT_TAG stable
)
FetchContent_MakeAvailable(zlib)
target_include_directories(zlib PUBLIC ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

# sqlitecpp
set(SQLITECPP_RUN_CPPLINT OFF CACHE BOOL "" FORCE)
set(SQLITECPP_RUN_CPPCHECK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  sqlitecpp
  GIT_REPOSITORY https://github.com/cuberite/SQLiteCpp.git
)
FetchContent_MakeAvailable(sqlitecpp)

# fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 7.1.2
)
FetchContent_MakeAvailable(fmt)

# spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.8.5
)
FetchContent_MakeAvailable(spdlog)

# spdlog
FetchContent_Declare(
  concurrencpp
  GIT_REPOSITORY https://github.com/David-Haim/concurrencpp.git
)
FetchContent_MakeAvailable(concurrencpp)

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
FetchContent_MakeAvailable(cppcodec)
target_include_directories(cppcodec PUBLIC ${cppcodec_SOURCE_DIR}/cppcodec)

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

find_package(CUDAToolkit)

target_link_libraries(librefrakt PRIVATE 
	xxhash 
	#CUDA::nvjpeg
	CUDA::nvrtc
	zlibstatic
	SQLiteCpp
	yaml-cpp
	cppcodec
	pugixml
	stb
)

target_link_libraries(librefrakt PUBLIC 
	spdlog 
	CUDA::cuda_driver
	fmt::fmt
	concurrencpp
	inja
)