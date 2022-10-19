add_executable(refraktor src/refraktor.cpp)

set_target_properties(refraktor PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

FetchContent_Declare(
  eznve
  GIT_REPOSITORY https://github.com/untrioctium/eznve.git
  GIT_TAG dev
)
FetchContent_MakeAvailable(eznve)

target_link_libraries(refraktor 
	PRIVATE 
	librefrakt 
	fmt::fmt
    eznve)