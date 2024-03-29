add_executable(refrakt-server src/refrakt-server.cpp)

set_target_properties(refrakt-server PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(refrakt-server PROPERTIES FIRST_PARTY true)

target_link_libraries(refrakt-server
	PRIVATE 
	librefrakt 
	spdlog
	concurrencpp)
	
#libuv
set(LIBUV_BUILD_SHARED OFF CACHE BOOL "")
set(LIBUV_BUILD_TESTS OFF CACHE BOOL "")
set(LIBUV_BUILD_BENCH OFF CACHE BOOL "")
FetchContent_Declare(
  libuv
  GIT_REPOSITORY https://github.com/libuv/libuv.git
  GIT_TAG v1.48.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libuv)

#uWebSockets
FetchContent_Declare(
  uws
  GIT_REPOSITORY https://github.com/uNetworking/uWebSockets.git
  GIT_TAG v20.62.0
)
FetchContent_MakeAvailable(uws)

file(GLOB_RECURSE US_SOURCES ${uws_SOURCE_DIR}/uSockets/src/*.c ${uws_SOURCE_DIR}/uSockets/src/*.cpp)
add_library(uSockets STATIC ${US_SOURCES})
target_include_directories(uSockets PUBLIC ${uws_SOURCE_DIR}/uSockets/src/)
target_compile_definitions(uSockets PRIVATE LIBUS_USE_LIBUV)
target_compile_definitions(uSockets PRIVATE LIBUS_NO_SSL)
target_link_libraries(uSockets PRIVATE uv_a)

target_link_libraries(refrakt-server PUBLIC uSockets)
target_include_directories(refrakt-server PRIVATE ${uws_SOURCE_DIR}/src)
target_compile_definitions(refrakt-server PRIVATE UWS_NO_ZLIB)

# inja, note that this also includes nlohmann::json
FetchContent_Declare(
 inja
 GIT_REPOSITORY https://github.com/pantor/inja.git
 GIT_TAG v3.4.0
 GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(inja)
target_link_libraries(refrakt-server PRIVATE inja)

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY    https://github.com/cameron314/readerwriterqueue
  GIT_TAG 8e7627d18c2108aca178888d88514179899a044f
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(readerwriterqueue)
target_link_libraries(refrakt-server PRIVATE readerwriterqueue)
