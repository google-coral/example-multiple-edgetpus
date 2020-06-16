# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
SHELL := /bin/bash
PYTHON3 ?= python3
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
PY3_VER ?= $(shell $(PYTHON3) -c "import sys;print('%d%d' % sys.version_info[:2])")
OS := $(shell uname -s)

# Allowed CPU values: k8, armv7a, aarch64, darwin
ifeq ($(OS),Linux)
CPU ?= k8
else
$(error $(OS) is not supported)
endif
ifeq ($(filter $(CPU),k8),)
$(error CPU must be k8)
endif

# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= dbg
ifeq ($(filter $(COMPILATION_MODE),opt dbg),)
$(error COMPILATION_MODE must be opt or dbg)
endif

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS_Linux := --crosstool_top=@crosstool//:toolchains \
                           --compiler=gcc \
                           --linkopt=-l:libedgetpu.so.1
BAZEL_BUILD_FLAGS_Darwin := --linkopt=-ledgetpu.1

ifeq ($(COMPILATION_MODE), opt)
BAZEL_BUILD_FLAGS_Linux += --linkopt=-Wl,--strip-all
endif

ifeq ($(CPU),k8)
BAZEL_BUILD_FLAGS_Linux += --copt=-includeglibc_compat.h
SWIG_WRAPPER_NAME := _edgetpu_cpp_wrapper.cpython-$(PY3_VER)m-x86_64-linux-gnu.so
endif

BAZEL_BUILD_FLAGS := --compilation_mode=$(COMPILATION_MODE) \
                     --copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
                     --verbose_failures \
                     --sandbox_debug \
                     --subcommands \
                     --define PY3_VER=$(PY3_VER) \
                     --cpu=$(CPU) \
                     --linkopt=-L$(MAKEFILE_DIR)/libedgetpu/direct/$(CPU) \
                     --experimental_repo_remote_exec
BAZEL_BUILD_FLAGS += $(BAZEL_BUILD_FLAGS_$(OS))

BAZEL_QUERY_FLAGS := --experimental_repo_remote_exec

# $(1): pattern, $(2) destination directory
define copy_out_files
pushd $(BAZEL_OUT_DIR); \
for f in `find . -name $(1) -type f`; do \
	mkdir -p $(2)/`dirname $$f`; \
	cp -f $(BAZEL_OUT_DIR)/$$f $(2)/$$f; \
done; \
popd
endef

EXAMPLES_OUT_DIR    := $(MAKEFILE_DIR)/out/$(CPU)/examples

# add make targets later
examples:
	bazel build $(BAZEL_BUILD_FLAGS) //src:multiple_edgetpu_demo
	mkdir -p $(EXAMPLES_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/multiple_edgetpu_demo \
	      $(EXAMPLES_OUT_DIR)

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/out \

DOCKER_WORKSPACE=$(MAKEFILE_DIR)
DOCKER_CPUS=k8
DOCKER_TAG_BASE=coral-edgetpu
include $(MAKEFILE_DIR)/docker/docker.mk

deb:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b

help:
	@echo "make all               - Build all C++ code"
	@echo "make tests             - Build all C++ tests"
	@echo "make benchmarks        - Build all C++ benchmarks"
	@echo "make tools             - Build all C++ tools"
	@echo "make examples          - Build all C++ examples"
	@echo "make swig              - Build python SWIG wrapper"
	@echo "make clean             - Remove generated files"
	@echo "make deb               - Build Debian packages for amd64"
	@echo "make deb-armhf         - Build Debian packages for armhf"
	@echo "make deb-arm64         - Build Debian packages for arm64"
	@echo "make wheel             - Build Python wheel"
	@echo "make help              - Print help message"

# Debugging util, print variable names. For example, `make print-ROOT_DIR`.
print-%:
	@echo $* = $($*)
