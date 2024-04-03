project(librefrakt LANGUAGES CXX)

set(HEADER_LIST 
	"include/librefrakt/traits/hashable.h"
	"include/librefrakt/traits/noncopyable.h"
	"include/librefrakt/util/hash.h"
	"include/librefrakt/util/cuda.h"
	"include/librefrakt/util/gpuinfo.h"
	"include/librefrakt/util/filesystem.h"
	"include/librefrakt/util/stb.h"
	"include/librefrakt/util/zlib.h"
	"include/librefrakt/util/color.h"
	"include/librefrakt/util/http.h"
	"include/librefrakt/util.h"
	"include/librefrakt/interface/jpeg_encoder.h"
	"include/librefrakt/interface/mp4_muxer.h"
	"include/librefrakt/interface/denoiser.h"
	"include/librefrakt/flame_types.h"
	"include/librefrakt/flame_info.h"
	"include/librefrakt/flame_compiler.h"
	"include/librefrakt/gpu_buffer.h"
	"include/librefrakt/image/tonemapper.h"
	"include/librefrakt/image/denoiser.h"
	"include/librefrakt/image/converter.h"
	"include/librefrakt/anima.h"
	"include/librefrakt/allocators.h"
	"include/librefrakt/factory.h"
	"include/librefrakt/constants.h"
)

add_library(librefrakt OBJECT
	"src/util/hash.cpp"
	"src/util/cuda.cpp"
	"src/util/filesystem.cpp"
	"src/util/gpuinfo.cpp"
	"src/util/zlib.cpp"
	"src/util/color.cpp"
	"src/util/stb.cpp"
	"src/impl/jpeg_encoder/nvjpeg.cpp"
	"src/impl/jpeg_encoder/cpu_jpeg_encoder.cpp"
	"src/impl/mp4_muxer/ffmpeg_muxer.cpp"
	"src/impl/denoiser/optix_denoiser.cpp"
	"src/impl/denoiser/oidn_denoiser.cpp"
	"src/impl/denoiser/null_denoiser.cpp"
	"src/util/http.cpp"
	"src/flame_types.cpp"
	"src/flame_info.cpp"
	"src/flame_compiler.cpp"
	"src/image/tonemapper.cpp"
	"src/image/denoiser.cpp"
	"src/image/converter.cpp"
	"src/anima.cpp"
	"src/allocators.cpp"
	${HEADER_LIST}
)

#set_property(TARGET librefrakt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(librefrakt PUBLIC include)
set_target_properties(librefrakt PROPERTIES FIRST_PARTY true)

# yaml-cpp
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_Declare(
  yaml_cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
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
target_include_directories(librefrakt PUBLIC ${sqlite3_SOURCE_DIR})

# glm
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm
  GIT_TAG 1.0.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)
target_link_libraries(librefrakt PUBLIC glm)

# concurrencpp
FetchContent_Declare(
  concurrencpp
  GIT_REPOSITORY https://github.com/David-Haim/concurrencpp.git
  GIT_TAG v.0.1.7
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(concurrencpp)

# stb
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG ae721c50eaf761660b4f90cc590453cdb0c2acd0
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
target_include_directories(librefrakt PRIVATE ${cppcodec_SOURCE_DIR}/cppcodec)

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
  GIT_TAG 88246d621abf7b6fba9332f49229d507f020e450
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(lua)

# sol3
FetchContent_Declare(
  sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG e8e122e9ce46f4f1c0b04003d8b703fe1b89755a
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(sol2)

# dylib
FetchContent_Declare(
    dylib
    GIT_REPOSITORY "https://github.com/martin-olivier/dylib"
    GIT_TAG c5b46c650a252bcda0dd94a102cdfc85907b9022
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dylib)

set(REPROC++ ON)
# reproc
FetchContent_Declare(
  reproc
  GIT_REPOSITORY https://github.com/DaanDeMeyer/reproc.git
  GIT_TAG 3179928ae7b085e41dfb846d987519fa7c12ffb3
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(reproc)

# intel open image denoise
FetchContent_Declare(
  oidn
  URL https://github.com/OpenImageDenoise/oidn/releases/download/v2.2.2/oidn-2.2.2.x64.windows.zip
  URL_HASH SHA256=5cc8bcc2a3321ef32547c3be70d43878a41324718cabdfb151b332a2a4928297
)
FetchContent_MakeAvailable(oidn)
include("${oidn_SOURCE_DIR}/lib/cmake/OpenImageDenoise-2.2.2/OpenImageDenoiseConfig.cmake")

# imath
#FetchContent_Declare(
#	imath
#	GIT_REPOSITORY "https://github.com/AcademySoftwareFoundation/imath"
#	GIT_TAG a3c1599c1084cc58416d7d4461eee4337d95b341
#	GIT_SHALLOW TRUE
#)
#FetchContent_MakeAvailable(imath)

# openexr
#set(OPENEXR_INSTALL OFF CACHE BOOL "" FORCE)
#set(OPENEXR_INSTALL_TOOLS OFF CACHE BOOL "" FORCE)
#set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
#set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
#set(OPENEXR_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
#set(OPENEXR_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
#FetchContent_Declare(
#    openexr
#    GIT_REPOSITORY "https://github.com/AcademySoftwareFoundation/openexr"
#    GIT_TAG        9d3e061a0b4090f268eee50e84fcccf5cbcad9ed
#	GIT_SHALLOW TRUE
#)
#FetchContent_MakeAvailable(openexr)



if(WIN32)
	set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
endif()

set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
set(HTTP_ONLY ON CACHE BOOL "" FORCE)
set(CURL_ENABLE_EXPORT_TARGET OFF)
set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  curl
  GIT_REPOSITORY https://github.com/curl/curl.git
  GIT_TAG a41cd15bec379fcea3b2a98e98c322061304b3fd
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(curl)

FetchContent_Declare(
  zip
  GIT_REPOSITORY https://github.com/kuba--/zip
  GIT_TAG 6f2116d77cdcbe544e4dbbf1e53895596ed14b89
  GIT_SHALLOW TRUE
)
FetchContent_Populate(zip)
add_library(zip ${zip_SOURCE_DIR}/src/zip.c ${zip_SOURCE_DIR}/src/zip.h ${zip_SOURCE_DIR}/src/miniz.h)
target_include_directories(zip PUBLIC ${zip_SOURCE_DIR}/src)


target_compile_definitions(librefrakt PUBLIC EZRTC_USE_ZLIB EZRTC_USE_SQLITE)

add_subdirectory(depend/flang)
add_subdirectory(depend/ezrtc)
add_subdirectory(depend/eznve)
add_subdirectory(depend/optix_downloader)
add_subdirectory(depend/roccu)

message(STATUS "Found Optix: $ENV{OPTIX_INCLUDE_DIRECTORY}")
target_include_directories(librefrakt PRIVATE $ENV{OPTIX_INCLUDE_DIRECTORY})

target_link_libraries(librefrakt PRIVATE 
	xxhash 
	#CUDA::nvjpeg
	stb
	zip
	CURL::libcurl_static
	OpenImageDenoise
	#OpenEXR::OpenEXR
)

target_link_libraries(librefrakt PUBLIC
	roccu
	flang
	ezrtc
	eznve
	spdlog 
	reproc++
	#CUDA::nvml
	#CUDA::cuda_driver
	concurrencpp
	inja
	lua_static
	sol2
	yaml-cpp
	Rpcrt4.lib
	sqlite3-static
	zlib
	pugixml
	dylib
)
