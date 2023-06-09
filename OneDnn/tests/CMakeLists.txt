include(FetchContent)

FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG devel
	GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Catch2)

mark_as_advanced(FETCHCONTENT_BASE_DIR)
mark_as_advanced(FETCHCONTENT_FULLY_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_QUIET)
mark_as_advanced(FETCHCONTENT_SOURCE_DIR_CATCH2)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED)
mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_CATCH2)

mark_as_advanced(CATCH_CONFIG_ANDROID_LOGWRITE)
mark_as_advanced(CATCH_CONFIG_BAZEL_SUPPORT)
mark_as_advanced(CATCH_CONFIG_COLOUR_WIN32)
mark_as_advanced(CATCH_CONFIG_CONSOLE_WIDTH)
mark_as_advanced(CATCH_CONFIG_COUNTER)
mark_as_advanced(CATCH_CONFIG_CPP11_TO_STRING)
mark_as_advanced(CATCH_CONFIG_CPP17_BYTE)
mark_as_advanced(CATCH_CONFIG_CPP17_OPTIONAL)
mark_as_advanced(CATCH_CONFIG_CPP17_STRING_VIEW)
mark_as_advanced(CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)
mark_as_advanced(CATCH_CONFIG_CPP17_VARIANT)
mark_as_advanced(CATCH_CONFIG_DEFAULT_REPORTER)
mark_as_advanced(CATCH_CONFIG_DISABLE)
mark_as_advanced(CATCH_CONFIG_DISABLE_EXCEPTIONS)
mark_as_advanced(CATCH_CONFIG_DISABLE_EXCEPTIONS_CUSTOM_HANDLER)
mark_as_advanced(CATCH_CONFIG_DISABLE_STRINGIFICATION)
mark_as_advanced(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS)
mark_as_advanced(CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER)
mark_as_advanced(CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER)
mark_as_advanced(CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER)
mark_as_advanced(CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER)
mark_as_advanced(CATCH_CONFIG_EXPERIMENTAL_REDIRECT)
mark_as_advanced(CATCH_CONFIG_FAST_COMPILE)
mark_as_advanced(CATCH_CONFIG_GETENV)
mark_as_advanced(CATCH_CONFIG_GLOBAL_NEXTAFTER)
mark_as_advanced(CATCH_CONFIG_NOSTDOUT)
mark_as_advanced(CATCH_CONFIG_NO_ANDROID_LOGWRITE)
mark_as_advanced(CATCH_CONFIG_NO_BAZEL_SUPPORT)
mark_as_advanced(CATCH_CONFIG_NO_COLOUR_WIN32)
mark_as_advanced(CATCH_CONFIG_NO_COUNTER)
mark_as_advanced(CATCH_CONFIG_NO_CPP11_TO_STRING)
mark_as_advanced(CATCH_CONFIG_NO_CPP17_BYTE)
mark_as_advanced(CATCH_CONFIG_NO_CPP17_OPTIONAL)
mark_as_advanced(CATCH_CONFIG_NO_CPP17_STRING_VIEW)
mark_as_advanced(CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS)
mark_as_advanced(CATCH_CONFIG_NO_CPP17_VARIANT)
mark_as_advanced(CATCH_CONFIG_NO_GETENV)
mark_as_advanced(CATCH_CONFIG_NO_GLOBAL_NEXTAFTER)
mark_as_advanced(CATCH_CONFIG_NO_POSIX_SIGNALS)
mark_as_advanced(CATCH_CONFIG_NO_USE_ASYNC)
mark_as_advanced(CATCH_CONFIG_NO_WCHAR)
mark_as_advanced(CATCH_CONFIG_NO_WINDOWS_SEH)
mark_as_advanced(CATCH_CONFIG_POSIX_SIGNALS)
mark_as_advanced(CATCH_CONFIG_PREFIX_ALL)
mark_as_advanced(CATCH_CONFIG_USE_ASYNC)
mark_as_advanced(CATCH_CONFIG_WCHAR)
mark_as_advanced(CATCH_CONFIG_WINDOWS_CRTDBG)
mark_as_advanced(CATCH_CONFIG_WINDOWS_SEH)
mark_as_advanced(CATCH_DEVELOPMENT_BUILD)
mark_as_advanced(CATCH_INSTALL_DOCS)
mark_as_advanced(CATCH_INSTALL_EXTRAS)

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS *.cpp)

set(CTEST_PARALLEL_LEVEL 4)
foreach(TEST_FILE ${TEST_FILES})
	get_filename_component(TEST_BIN ${TEST_FILE} NAME_WE)
	add_executable(${TEST_BIN} ${TEST_FILE})
	target_link_libraries(${TEST_BIN} Catch2WithMain ${DNNL_RUNTIME})
	add_test(${TEST_BIN} ${TEST_BIN})
	set_tests_properties(${TEST_BIN} PROPERTIES TIMEOUT 5)
endforeach()

# Add OneDNN examples as tests

function (dnnl_test TEST_NAME)
	find_file(TEST_SRC ${TEST_NAME}.cpp REQUIRED NO_DEFAULT_PATH PATHS ${CMAKE_BINARY_DIR}/InteloneDNN PATH_SUFFIXES examples)
	add_executable(${TEST_NAME} ${TEST_SRC})
	target_link_libraries(${TEST_NAME} ${DNNL_RUNTIME})
	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

dnnl_test(getting_started)
#dnnl_test(performance_profiling)
#dnnl_test(cnn_inference_f32)
