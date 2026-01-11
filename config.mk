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


# LIBRARIES
STB_CFLAGS:=-I$(DEPS_DIR)/src/stb
OPENSSL_LDFLAGS := -lssl -lcrypto
GLAD_LDFLAGS := -lglad
OPENGL_LDFLAGS := $(GLAD_LDFLAGS) -lGL
FREETYPE_CFLAGS := $(shell pkg-config --cflags freetype2)
FREETYPE_LDFLAGS := $(shell pkg-config --libs freetype2)
BLAS_LDFLAGS := -lblas -llapack -llapacke
SUITESPARSE_LDFLAGS := -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse
CERES_CFLAGS := -I$(DEPS_DIR)/include/eigen3
CERES_LDFLAGS := -lgflags -lglog -lceres
ASSIMP_LDFLAGS := -lassimp
APRILTAG_LDFLAGS := -lapriltag
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
	-I$(DEPS_DIR)/glad \
	-fPIC \
	$(STB_CFLAGS) \
	$(FREETYPE_CFLAGS) \
	$(CERES_CFLAGS)


# LDFLAGS
RPATH := -Wl,-rpath,$(DEPS_DIR)/lib
LDFLAGS= \
	-ggdb \
	-L$(DEPS_DIR)/lib \
	$(RPATH) \
	$(XYZ_LDFLAGS) \
	$(CERES_LDFLAGS) \
	$(APRILTAG_LDFLAGS) \
	$(OPENGL_LDFLAGS) \
	$(SUITESPARSE_LDFLAGS) \
	$(BLAS_LDFLAGS) \
	$(OPENSSL_LDFLAGS) \
	$(ASSIMP_LDFLAGS) \
	$(YAML_LDFLAGS) \
	$(FREETYPE_LDFLAGS) \
	-lglfw \
	-lstdc++ \
	-lpthread \
	-lm \
	-ldl


# ARCHIVER SETTTINGS
AR = ar
ARFLAGS = rvs


# TARGETS
LIBXYZ := $(BLD_DIR)/libxyz.a
LIBXYZ_OBJS := \
	$(BLD_DIR)/xyz.o


# TESTS
TESTS := \
	$(BLD_DIR)/test_xyz \
	$(BLD_DIR)/test_gui
	# $(BLD_DIR)/test_ceres \
	# $(BLD_DIR)/test_se
