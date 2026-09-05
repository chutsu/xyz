# Benchmark xyz's image_harris()/image_good_features() vs OpenCV's
# cornerHarris()/goodFeaturesToTrack().
#
# Usage:
#   make            build and run
#   make build      build only
#   make run        run the benchmark
#   make clean      remove build artifacts
#
# The xyz library is (re)built in release mode (via make _libxyz_internal
# BUILD_TYPE=release) for a fair comparison. Note: this leaves libxyz.a in
# release form until the caller restores the default debug build.
ROOT := $(abspath ../..)

# OpenCV v4 (for cornerHarris / goodFeaturesToTrack) and FreeType (needed by the
# xyz headers and by OpenCV's freetype module).
OpenCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OpenCV_LDFLAGS := $(shell pkg-config --libs opencv4)
FREETYPE_CFLAGS := $(shell pkg-config --cflags freetype2)
FREETYPE_LDFLAGS := $(shell pkg-config --libs freetype2)

# Build flags. config.mk forces BUILD_TYPE := debug (which would add ASAN), so
# the C wrapper is compiled with explicit release flags instead. libxyz.a is
# rebuilt in release mode by the rule below regardless.
include $(ROOT)/config.mk

CC := clang
CXX := clang++

# bench_xyz.c is plain C calling the xyz C API.
CWRAP_CFLAGS := -std=c99 -O3 -march=native -fopenmp -DNDEBUG -Wall \
	-I$(ROOT) -I$(ROOT)/deps/include -I$(ROOT)/deps/src/glad \
	-I$(ROOT)/deps/src/stb $(FREETYPE_CFLAGS)

# bench.cpp is C++ and uses OpenCV.
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -DNDEBUG -Wall \
	$(OpenCV_CFLAGS) -I$(ROOT)

# Link with clang++ so the OpenMP (__kmpc_*) symbols emitted by clang are
# resolved by the LLVM libomp runtime. libxyz.a pulls in xyz's external deps.
LDLIBS := $(OpenCV_LDFLAGS) -L$(ROOT)/$(BLD_DIR) -lxyz -fopenmp \
	-L$(ROOT)/deps/lib -Wl,-rpath,$(ROOT)/deps/lib \
	-lapriltag -lglad -lGL -lglfw \
	-llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lcxsparse \
	-lblas -llapack -llapacke \
	-lssl -lcrypto -lassimp -lyaml \
	$(FREETYPE_LDFLAGS) \
	-lstdc++ -lpthread -lm -ldl

BLD := $(ROOT)/$(BLD_DIR)
ALL_OBJS := $(BLD)/bench_xyz.o $(BLD)/bench_corners.o
TARGET := $(BLD)/bench_corners

.PHONY: all build run clean

all: build run

build: $(TARGET)

# Build a release libxyz into build/libxyz.a. Force a recompile of xyz.o, since
# make alone won't rebuild it when BUILD_TYPE flips between debug and release.
$(ROOT)/$(BLD_DIR)/libxyz.a: FORCE
	@rm -f $(ROOT)/$(BLD_DIR)/xyz.o $(ROOT)/$(BLD_DIR)/libxyz.a
	@$(MAKE) -C $(ROOT) _libxyz_internal BUILD_TYPE=release --no-print-directory

$(BLD)/bench_xyz.o: bench_xyz.c bench_xyz.h
	@mkdir -p $(BLD)
	$(CC) $(CWRAP_CFLAGS) -c $< -o $@

$(BLD)/bench_corners.o: bench.cpp bench_xyz.h
	@mkdir -p $(BLD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(ALL_OBJS) $(ROOT)/$(BLD_DIR)/libxyz.a
	$(CXX) $(CXXFLAGS) $(ALL_OBJS) -o $@ $(LDLIBS)

run: build
	@$(TARGET) 640 480

FORCE:

clean:
	@rm -f $(TARGET) $(ALL_OBJS)
