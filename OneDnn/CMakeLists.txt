cmake_minimum_required(VERSION 3.6)

project(CarvDNN)

include(CTest)
include(cmake/arax.cmake)
include(cmake/onednn.cmake)
include(cmake/Coverage.cmake)
include(cmake/tensorflow.cmake)

set(DNNL_RUNTIME
	dnnl_arax
	CACHE STRING "DNNL Runtime to use"
)
set_property(CACHE DNNL_RUNTIME PROPERTY STRINGS dnnl_arax dnnl_native)

option(TRACE_CALLS "Trace DNNL calls" OFF)

aux_source_directory(src SRC)

add_library(dnnl_arax SHARED ${SRC})
target_link_libraries(dnnl_arax PUBLIC arax)
target_include_directories(dnnl_arax PUBLIC ${DNNL_INCLUDE_DIRS})
target_compile_definitions(dnnl_arax PRIVATE -DARAX_WRAPPERS)

add_library(dnnl_cntrl SHARED ${SRC})
target_link_libraries(dnnl_cntrl PUBLIC arax dnnl_native)
target_compile_definitions(dnnl_cntrl PRIVATE -DARAX_HANDLERS)
set_target_properties(dnnl_cntrl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cntrl)

if(TRACE_CALLS)
	target_compile_definitions(dnnl_arax PRIVATE -DTRACE_CALLS)
	target_compile_definitions(dnnl_cntrl PRIVATE -DTRACE_CALLS)
endif()

add_subdirectory(example)
add_subdirectory(tests)

add_subdirectory(tf)
