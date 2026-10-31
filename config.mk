# PATHS
MKFILE_PATH=$(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(shell pwd)
BLD_DIR := build
INC_DIR := $(realpath $(PWD))
DEPS_DIR := $(realpath deps)
PREFIX := /opt/xyz
PYTHON3_PATH := $(shell python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(DEPS_DIR)/lib

# SETTINGS
BUILD_TYPE := debug
# BUILD_TYPE := release
CI_MODE := 0
CC := clang
# CC := gcc
# CC := tcc
CXX := clang++


# LIBRARIES
# (GLAD/GLFW/assimp/freetype are NOT needed here -- they moved to the
# self-contained opengl/ module along with the code that used them)
STB_CFLAGS:=-I$(DEPS_DIR)/src/stb
OPENSSL_LDFLAGS := -lssl -lcrypto
BLAS_LDFLAGS := -lblas -llapack -llapacke
SUITESPARSE_LDFLAGS := -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse
APRILTAG_LDFLAGS := -lapriltag
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LDFLAGS := $(shell pkg-config --libs opencv4)
YAML_LDFLAGS := -lyaml
XYZ_LDFLAGS := -L$(BLD_DIR) -lxyz


# CFLAGS
CFLAGS := -std=c99 \
	-D_POSIX_C_SOURCE=200809L \
	-D_GNU_SOURCE \
	-Wall \
	-Wpedantic \
	-Wstrict-prototypes \
	-ggdb

ifeq ($(BUILD_TYPE), debug)
	CFLAGS += -g -fopenmp
	ADDRESS_SANITIZER := 1
else
	CFLAGS += -g -O3 -march=native -DNDEBUG -fopenmp
	ADDRESS_SANITIZER := 0
endif

ifeq ($(ADDRESS_SANITIZER), 1)
ifeq ($(CC), gcc)
	CFLAGS += -fsanitize=address -static-libasan
else
	CFLAGS += -fsanitize=address -static-libsan
endif
endif

ifeq ($(CI_MODE), 1)
	CFLAGS += -DMU_REDIRECT_STREAMS=1 -DCI_MODE=1
endif


CFLAGS += \
	-I$(INC_DIR) \
	-I$(DEPS_DIR)/include \
	-fPIC \
	$(STB_CFLAGS)


# CXXFLAGS (benchmark_xyz.cpp: C++ wrapper around xyz + OpenCV comparisons.
# Timing is meaningless under ASan/debug, so prefer `make BUILD_TYPE=release
# benchmark`.)
CXXFLAGS := -std=c++17 -Wall -ggdb

ifeq ($(BUILD_TYPE), debug)
	CXXFLAGS += -g -fopenmp
else
	CXXFLAGS += -g -O3 -march=native -DNDEBUG -fopenmp
endif

CXXFLAGS += \
	-I$(INC_DIR) \
	-I$(DEPS_DIR)/include \
	-fPIC \
	$(STB_CFLAGS) \
	$(OPENCV_CFLAGS)


# LDFLAGS
RPATH := -Wl,-rpath,$(DEPS_DIR)/lib
LDFLAGS= \
	-ggdb \
	-L$(DEPS_DIR)/lib \
	$(RPATH) \
	$(XYZ_LDFLAGS) \
	$(APRILTAG_LDFLAGS) \
	$(SUITESPARSE_LDFLAGS) \
	$(BLAS_LDFLAGS) \
	$(OPENSSL_LDFLAGS) \
	$(YAML_LDFLAGS) \
	-lstdc++ \
	-lpthread \
	-lm \
	-ldl

# CXXLDFLAGS (benchmark_xyz.cpp only, adds OpenCV on top of LDFLAGS)
CXXLDFLAGS = $(LDFLAGS) $(OPENCV_LDFLAGS)


# ARCHIVER SETTTINGS
AR = ar
ARFLAGS = rvs


# TARGETS
LIBXYZ := $(BLD_DIR)/libxyz.a
LIBXYZ_OBJS := $(BLD_DIR)/xyz.o


# TESTS
# (the opengl/ module is self-contained -- see opengl/Makefile)
TESTS := $(BLD_DIR)/test_xyz

# BENCHMARKS
BENCHMARKS := $(BLD_DIR)/benchmark_xyz
