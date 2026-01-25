project(librefrakt LANGUAGES CXX)

set(HEADER_LIST 
	"include/librefrakt/util/gpuinfo.hpp"
	"include/librefrakt/util/filesystem.hpp"
	"include/librefrakt/interface/jpeg_encoder.hpp"
	"include/librefrakt/interface/mp4_muxer.hpp"
	"include/librefrakt/interface/denoiser.hpp"
	"include/librefrakt/flame_compiler.hpp"
	"include/librefrakt/image/tonemapper.hpp"
	"include/librefrakt/image/converter.hpp"
	"include/librefrakt/allocators.hpp"
	"include/librefrakt/constants.hpp"
	"include/librefrakt/util/cuda.hpp"
)

add_library(librefrakt STATIC
	"src/util/filesystem.cpp"
	"src/util/gpuinfo.cpp"
	"src/flame_compiler.cpp"
	"src/image/tonemapper.cpp"
	"src/image/converter.cpp"
	"src/allocators.cpp"
	"src/util/cuda.cpp"
	${HEADER_LIST}
)

#set_property(TARGET librefrakt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(librefrakt PUBLIC include)
set_target_properties(librefrakt PROPERTIES FIRST_PARTY true)

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



#if(WIN32)
#	set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
#endif()

#set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
#set(HTTP_ONLY ON CACHE BOOL "" FORCE)
#set(CURL_ENABLE_EXPORT_TARGET OFF)
#set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
#set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
#set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)

#FetchContent_Declare(
#  curl
#  GIT_REPOSITORY https://github.com/curl/curl.git
#  GIT_SHALLOW TRUE
#)
#FetchContent_MakeAvailable(curl)

#FetchContent_Declare(
#  zip
#  GIT_REPOSITORY https://github.com/kuba--/zip
#  GIT_SHALLOW TRUE
#)
#FetchContent_Populate(zip)
#add_library(zip ${zip_SOURCE_DIR}/src/zip.c ${zip_SOURCE_DIR}/src/zip.h ${zip_SOURCE_DIR}/src/miniz.h)
#target_include_directories(zip PUBLIC ${zip_SOURCE_DIR}/src)


target_compile_definitions(librefrakt PUBLIC EZRTC_USE_ZLIB EZRTC_USE_SQLITE)

add_subdirectory(depend/ezrtc)
add_subdirectory(depend/eznve)
add_subdirectory(depend/optix_downloader)
add_subdirectory(depend/roccu)

target_link_libraries(librefrakt PUBLIC
	librefrakt-core
	roccu
 	ezrtc
	eznve
	Rpcrt4.lib
)

add_library(librefrakt-impl OBJECT
	"src/impl/jpeg_encoder/nvjpeg.cpp"
	"src/impl/jpeg_encoder/cpu_jpeg_encoder.cpp"
	"src/impl/mp4_muxer/ffmpeg_muxer.cpp"
	"src/impl/denoiser/optix_denoiser.cpp"
	"src/impl/denoiser/oidn_denoiser.cpp"
	"src/impl/denoiser/null_denoiser.cpp"
)

# dylib
FetchContent_Declare(
    dylib
    GIT_REPOSITORY "https://github.com/martin-olivier/dylib"
	GIT_TAG v2.2.1
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dylib)

set(REPROC++ ON)
# reproc
FetchContent_Declare(
  reproc
  GIT_REPOSITORY https://github.com/DaanDeMeyer/reproc.git
  GIT_TAG main
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

target_link_libraries(librefrakt-impl PRIVATE 
	OpenImageDenoise
	reproc++
	dylib
)

message(STATUS "Found Optix: $ENV{OPTIX_INCLUDE_DIRECTORY}")
target_include_directories(librefrakt-impl PRIVATE $ENV{OPTIX_INCLUDE_DIRECTORY})

target_link_libraries(librefrakt-impl PUBLIC librefrakt)