# opengl: OpenGL/GLFW rendering module (math/shaders/camera, GUI window
# and input handling, 3D drawables). Builds on top of libxyz.a, but is
# otherwise self-contained -- it builds its own libglad.a here rather
# than depending on the root for it, and is NOT part of the root
# Makefile's `make libxyz`/`make tests`.
#
# Usage:
#   cd .. && make libxyz          # libxyz.a must exist first
#   cd opengl && make test        # build + run test_opengl
#
# Flag VALUES below are duplicated from ../config.mk rather than done
# via `include ../config.mk`, because that file's INC_DIR/DEPS_DIR/
# BLD_DIR are computed from $(shell pwd) and so resolve to the wrong
# directory when make runs from here. Keep the library flags (GLFW,
# assimp, freetype, BLAS/SuiteSparse/yaml/openssl -- the latter three
# needed transitively, since libxyz.a is currently one single xyz.o
# archive member and linking any symbol from it pulls in all of it) in
# sync with ../config.mk if that file's dependency list changes.

ROOT_DIR := ..
SRC_DIR := $(ROOT_DIR)/src
DEPS_DIR := $(ROOT_DIR)/deps
ROOT_BLD_DIR := $(ROOT_DIR)/build
BLD_DIR := $(ROOT_BLD_DIR)/opengl

CC := clang
BUILD_TYPE := debug
CI_MODE := 0

CFLAGS := -std=c99 \
	-D_POSIX_C_SOURCE=200809L \
	-D_GNU_SOURCE \
	-Wall \
	-Wpedantic \
	-Wstrict-prototypes \
	-ggdb

ifeq ($(BUILD_TYPE), debug)
	CFLAGS += -g -fopenmp -fsanitize=address -static-libsan
else
	CFLAGS += -g -O3 -march=native -DNDEBUG -fopenmp
endif

ifeq ($(CI_MODE), 1)
	CFLAGS += -DMU_REDIRECT_STREAMS=1 -DCI_MODE=1
endif

CFLAGS += \
	-I$(SRC_DIR) \
	-I$(DEPS_DIR)/include \
	-I$(DEPS_DIR)/src/glad \
	-I$(DEPS_DIR)/src/stb \
	-fPIC \
	$(shell pkg-config --cflags freetype2)

LDFLAGS := \
	-ggdb \
	-L$(ROOT_BLD_DIR) -lxyz \
	-L$(BLD_DIR) -lglad \
	-L$(DEPS_DIR)/lib \
	-Wl,-rpath,$(realpath $(DEPS_DIR))/lib \
	-lGL -lglfw -lassimp \
	$(shell pkg-config --libs freetype2) \
	-llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse \
	-lblas -llapacke \
	-lapriltag \
	-lssl -lcrypto \
	-lyaml \
	-lstdc++ \
	-lpthread \
	-lm \
	-ldl

AR := ar
ARFLAGS := rvs

.PHONY: help
help: ## List make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;34m%-12s\033[0m%s\n", $$1, $$2}'

$(BLD_DIR):
	@mkdir -p $(BLD_DIR)

$(BLD_DIR)/libglad.a: | $(BLD_DIR)
	@echo "CC [glad.c]"
	@$(CC) -I$(DEPS_DIR)/src/glad -c $(DEPS_DIR)/src/glad/glad.c \
		-o $(BLD_DIR)/glad.o
	@$(AR) $(ARFLAGS) $@ $(BLD_DIR)/glad.o > /dev/null 2>&1

$(BLD_DIR)/opengl.o: opengl.c opengl.h | $(BLD_DIR)
	@echo "CC [opengl.c]"
	@$(CC) $(CFLAGS) -c opengl.c -o $@

$(BLD_DIR)/libopengl.a: $(BLD_DIR)/opengl.o
	@echo "AR [libopengl.a]"
	@$(AR) $(ARFLAGS) $@ $(BLD_DIR)/opengl.o > /dev/null 2>&1

.PHONY: libopengl
libopengl: $(BLD_DIR)/libglad.a $(BLD_DIR)/libopengl.a ## Build libopengl.a

$(BLD_DIR)/test_opengl: test_opengl.c $(BLD_DIR)/libopengl.a $(BLD_DIR)/libglad.a
	@echo "TEST [test_opengl]"
	@$(CC) $(CFLAGS) test_opengl.c -o $@ -L$(BLD_DIR) -lopengl $(LDFLAGS)

.PHONY: test
test: libopengl $(BLD_DIR)/test_opengl ## Build and run test_opengl
	@$(BLD_DIR)/test_opengl

.PHONY: cppcheck
cppcheck: ## Run cppcheck
	@cppcheck -I$(SRC_DIR) opengl.c opengl.h

.PHONY: clean
clean: ## Clean this module's build output
	@rm -rf $(BLD_DIR)
