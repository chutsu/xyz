# PATHS
CUR_DIR := $(shell pwd)
BLD_DIR := build
INC_DIR := $(realpath $(PWD))
DEPS_DIR := $(realpath ../third_party)
PREFIX := /opt/xyz
PYTHON3_PATH := $(shell python3 -c "import site; print(site.getsitepackages()[0])")


# COMPILER SETTINGS
BUILD_TYPE := debug
# BUILD_TYPE := release
ADDRESS_SANITIZER := 1
# ADDRESS_SANITIZER := 0
# CC := clang
CC := gcc
# CC := tcc

# LIBRARIES
STB_CFLAGS:=-I$(DEPS_DIR)/src/stb
OPENSSL_LDFLAGS := -lssl -lcrypto

GLEW_LDFLAGS := -lGLEW
SDL2_CFLAGS:=$(shell sdl2-config --cflags)
SDL2_LDFLAGS := $(shell sdl2-config --libs) -lSDL2_image
OPENGL_LDFLAGS := $(SDL2_LDFLAGS) $(GLEW_LDFLAGS) -lGL

BLAS_LDFLAGS := -lblas -llapack -llapacke
SUITESPARSE_LDFLAGS := -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse
CERES_CFLAGS := -I/usr/include/eigen3
CERES_LDFLAGS := -lgflags -lglog -lceres

ASSIMP_LDFLAGS := -lassimp

APRILTAG_LDFLAGS := -L$(DEPS_DIR)/lib -lapriltag
YAML_LDFLAGS := -lyaml
XYZ_LDFLAGS := -L$(BLD_DIR) -lxyz


# CFLAGS
CFLAGS :=

ifeq ($(BUILD_TYPE), debug)
	CFLAGS += -g -Wall -Wpedantic -fopenmp
else
	CFLAGS += -g -O2 -march=native -mtune=native -Wall -Wpedantic -DNDEBUG -fopenmp
endif

ifeq ($(ADDRESS_SANITIZER), 1)
ifeq ($(CC), gcc)
  CFLAGS += -fsanitize=address -static-libasan
else
  CFLAGS += -fsanitize=address -static-libsan
endif
endif

CFLAGS += \
	-I$(INC_DIR) \
	-I$(DEPS_DIR)/include \
	-fPIC \
	$(SDL2_CFLAGS) \
	$(STB_CFLAGS) \
	$(CERES_CFLAGS)


# LDFLAGS
RPATH := -Wl,-rpath,$(DEPS_DIR)/lib
LDFLAGS= \
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
	-lglfw3 \
	-lstdc++ \
	-lpthread \
  -lm


# ARCHIVER SETTTINGS
AR = ar
ARFLAGS = rvs


# TARGETS
LIBXYZ := $(BLD_DIR)/libxyz.a
LIBXYZ_OBJS := \
	$(BLD_DIR)/xyz.o \
	$(BLD_DIR)/aprilgrid.o \
	$(BLD_DIR)/euroc.o \
	$(BLD_DIR)/gui.o \
	$(BLD_DIR)/sbgc.o \
	$(BLD_DIR)/ceres_bridge.o

TESTS := \
	test_xyz \
	test_aprilgrid \
	test_sbgc \
	test_ubx

.PHONY: benchmarks build scripts shaders test_data viz


help:
	@echo "\033[1;34m[make targets]:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;36m%-20s\033[0m%s\n", $$1, $$2}'

$(BLD_DIR)/%.o: %.c %.h Makefile
	@echo "CC [$<]"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BLD_DIR)/ceres_bridge.o: ceres_bridge.cpp Makefile
	@echo "CXX [ceres_bridge.c]"
	@g++ -Wall -O3 -c ceres_bridge.cpp -o $(BLD_DIR)/ceres_bridge.o -I/usr/include/eigen3

$(BLD_DIR)/libxyz.a: $(LIBXYZ_OBJS)
	@echo "AR [libxyz.a]"
	@$(AR) $(ARFLAGS) \
		$(BLD_DIR)/libxyz.a \
		$(LIBXYZ_OBJS) \
		> /dev/null 2>&1

setup:
	@mkdir -p $(BLD_DIR)
	@cp -r shaders $(BLD_DIR)
	@cp -r test_data $(BLD_DIR)

clean:  ## Clean build directory
	@rm -rf $(BLD_DIR)

libxyz: setup $(BLD_DIR)/libxyz.a  ## Compile libxyz

install: ## Install libxyz
	mkdir -p $(PREFIX)
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	ln -sf $(CUR_DIR)/build/libxyz.a $(PREFIX)/lib/libxyz.a
	ln -sf $(CUR_DIR)/aprilgrid.h $(PREFIX)/include/aprilgrid.h
	ln -sf $(CUR_DIR)/ceres_bridge.h $(PREFIX)/include/ceres_bridge.h
	ln -sf $(CUR_DIR)/euroc.h $(PREFIX)/include/euroc.h
	ln -sf $(CUR_DIR)/gui.h $(PREFIX)/include/gui.h
	ln -sf $(CUR_DIR)/http.h $(PREFIX)/include/http.h
	ln -sf $(CUR_DIR)/xyz.h $(PREFIX)/include/xyz.h
	ln -sf $(CUR_DIR)/sbgc.h $(PREFIX)/include/sbgc.h
	ln -sf $(CUR_DIR)/stb_image.h $(PREFIX)/include/stb_image.h
	ln -sf $(CUR_DIR)/xyz.py $(PYTHON3_PATH)/xyz.py

uninstall: ## Uninstall libxyz
	rm $(PREFIX)/lib/libxyz.a
	rm $(PREFIX)/include/aprilgrid.h
	rm $(PREFIX)/include/euroc.h
	rm $(PREFIX)/include/gui.h
	rm $(PREFIX)/include/http.h
	rm $(PREFIX)/include/xyz.h
	rm $(PREFIX)/include/sbgc.h
	rm $(PREFIX)/include/stb_image.h
	rm $(PYTHON3_PATH)/xyz.py

avs: $(BLD_DIR)/libxyz.a
	@g++ \
		-std=c++17 \
		-fsanitize=address -static-libasan \
		-fopenmp \
		-g \
		-I$(DEPS_DIR)/include \
		-I/usr/include/eigen3 \
		$(shell pkg-config opencv4 --cflags) \
		avs.cpp \
		-o $(BLD_DIR)/avs \
		$(LDFLAGS) \
		-lxyz \
		$(shell pkg-config opencv4 --libs)

test_xyz: libxyz  ## Compile test_xyz
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_xyz.c -o $(BLD_DIR)/test_xyz $(LDFLAGS)
	./build/test_xyz --target $(TEST_TARGET)

test_aprilgrid:  ## Compile test_aprilgrid
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_aprilgrid.c -o $(BLD_DIR)/test_aprilgrid $(LDFLAGS)
	@./build/test_aprilgrid

test_euroc:  ## Compile test_euroc
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_euroc.c -o $(BLD_DIR)/test_euroc $(LDFLAGS)
	@./build/test_euroc

test_gui:  ## Compile test_gui
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_gui.c -o $(BLD_DIR)/test_gui $(LDFLAGS)

test_sbgc:  ## Compile test_sbgc
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_sbgc.c -o $(BLD_DIR)/test_sbgc $(LDFLAGS)
	@./build/test_sbgc

test_ubx:  ## Compile test_ubx
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_ubx.c -o $(BLD_DIR)/test_ubx $(LDFLAGS)
	@./build/test_ubx

test_http:  ## Compile test_http
	@echo "CC [$@]"
	@$(CC) $(CFLAGS) test_http.c -o $(BLD_DIR)/test_http $(LDFLAGS)
	@./build/test_http

tests: test_xyz test_aprilgrid test_sbgc test_ubx

ci:
	@$(CC) $(CFLAGS) -DMU_REDIRECT_STREAMS=1 test_xyz.c -o $(BLD_DIR)/test_xyz $(LDFLAGS)
	@./build/test_xyz
