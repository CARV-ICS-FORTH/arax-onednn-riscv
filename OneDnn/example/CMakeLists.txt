add_executable(example example.cpp)

target_link_libraries(example ${DNNL_RUNTIME})

add_custom_target(run
        COMMAND ./example/example
        DEPENDS example
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(gdb
        COMMAND gdb ./example/example
        DEPENDS example
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
