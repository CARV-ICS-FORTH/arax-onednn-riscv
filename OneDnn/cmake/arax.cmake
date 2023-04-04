find_package(arax QUIET)

if (NOT arax_DIR)
	get_filename_component(ARAX_DEF_BUILD_PATH "../arax/build" ABSOLUTE)
	set(ARAX_BUILD_PATH ${ARAX_DEF_BUILD_PATH} CACHE STRING "Arax build location")
	include("${ARAX_BUILD_PATH}/arax.cmake")
endif()
