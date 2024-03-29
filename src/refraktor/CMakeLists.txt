add_executable(refraktor src/refraktor.cpp)

set_target_properties(refraktor PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(refraktor PROPERTIES FIRST_PARTY true)

target_link_libraries(refraktor 
	PRIVATE 
	librefrakt 
)