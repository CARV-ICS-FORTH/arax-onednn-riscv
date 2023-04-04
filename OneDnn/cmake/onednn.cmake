include(FetchContent)

FetchContent_Declare(
	OneDNN
	GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
	GIT_TAG v2.6.2
	GIT_SHALLOW true
	GIT_PROGRESS true
	CMAKE_ARGS -DCMAKE_BUILD_TYPE='DEBUG' -DDNNL_BUILD_TESTS=off
	INSTALL_COMMAND ""
	UPDATE_COMMAND ""
        PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/cmake/onednn_patch.patch
        SOURCE_DIR "InteloneDNN"
)

FetchContent_MakeAvailable(OneDNN)

set(DNNL_INCLUDE_DIRS ${onednn_BINARY_DIR}/include ${CMAKE_BINARY_DIR}/InteloneDNN/include ${CMAKE_BINARY_DIR}/InteloneDNN/src)

add_library(dnnl_native INTERFACE)
target_include_directories(dnnl_native INTERFACE ${DNNL_INCLUDE_DIRS})
target_link_libraries(dnnl_native INTERFACE dnnl)
