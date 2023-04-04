include(ExternalProject)

find_program(PY_EXE python)

ExternalProject_Add(
	bazel
	URL https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-linux-x86_64
	DOWNLOAD_DIR ${CMAKE_BINARY_DIR}
	DOWNLOAD_NAME ${CMAKE_BINARY_DIR}/bazel_exe
	DOWNLOAD_NO_EXTRACT true
        EXCLUDE_FROM_ALL true
	CONFIGURE_COMMAND chmod 755 ${CMAKE_BINARY_DIR}/bazel_exe
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
)

# file(DOWNLOAD https://github.com/bazelbuild/bazel/releases/download/4.2.1/bazel-4.2.1-linux-x86_64 ${CMAKE_BINARY_DIR}/bazel_exe)
# file(CHMOD ${CMAKE_BINARY_DIR}/bazel_exe PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

ExternalProject_Add(
	tensorflow_project
	GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
	GIT_TAG v2.8.0
	GIT_SHALLOW true
	GIT_PROGRESS true
	BUILD_IN_SOURCE true
	UPDATE_COMMAND ""
	PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/tf/0001-Dynamicaly-link-with-MKL-OneAPI.patch
	CONFIGURE_COMMAND TF_NEED_GCP=0 TF_NEED_HDFS=0 TF_NEED_S3=0 TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_NEED_MKL=1 CC_OPT_FLAGS=-mavx PYTHON_BIN_PATH=${PY_EXE} USE_DEFAULT_PYTHON_LIB_PATH=1 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=0 python configure.py
	BUILD_COMMAND ${CMAKE_BINARY_DIR}/bazel_exe build --config=opt --config=v2 tensorflow/tools/pip_package:build_pip_package
	INSTALL_COMMAND ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ${CMAKE_BINARY_DIR}/wheel --cpu
        EXCLUDE_FROM_ALL true
	DEPENDS bazel
)

add_custom_target(tensorflow DEPENDS tensorflow_project
	COMMENT "Install command: pip install --upgrade ${CMAKE_BINARY_DIR}/wheel/*.whl"
)

