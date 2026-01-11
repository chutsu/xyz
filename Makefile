include config.mk
.PHONY: benchmarks build docs scripts src deps tools

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;34m%-14s\033[0m%s\n", $$1, $$2}'

deps: ## Install dependencies
	@# Update apt
	@sudo apt-get update -qq

	@# Base dev tools
	@sudo apt-get install -y -q \
		build-essential \
		pkg-config \
		autoconf \
		make \
		cmake \
		git \
		mercurial \
		g++ \
		clang \
		tcc \
		vim \
		vifm

	@# Base packages
	@sudo apt-get install -y -q \
		libyaml-dev \
		libssl-dev \
		libfreetype-dev \
		libfreetype6 \
		libgl1-mesa-dev

	@# Linear algebra base
	@sudo apt-get install -y -q \
		libomp-dev \
		libmpfr-dev \
		libblas-dev \
		liblapack-dev \
		liblapacke-dev \
		libmetis-dev \
		libsuitesparse-dev \
		libceres-dev \
		libeigen3-dev

	@# Computer vision
	@sudo apt-get install -y -q \
		libopencv-dev \
		libapriltag-dev \

	@# Computer graphics base
	@sudo apt-get install -y -q \
		libx11-dev \
		libwayland-dev \
		libxkbcommon-dev \
		libxrandr-dev \
		libxinerama-dev \
		libxcursor-dev \
		libxi-dev \
		libassimp-dev \
		libglfw3-dev

docs: ## Build docs
	@cd docs && livereload .

setup:
	@mkdir -p $(BLD_DIR)
	@cp -r deps/fonts $(BLD_DIR)
	@cp -r src/test_data $(BLD_DIR)

clean:  ## Clean
	@rm -rf $(BLD_DIR)

$(BLD_DIR)/test_%: src/test_%.c libxyz
	@echo "TEST [$(notdir $@)]"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lxyz

$(BLD_DIR)/%.o: src/%.c src/%.h Makefile
	@echo "CC [$(notdir $<)]"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BLD_DIR)/xyz_ceres.o: src/xyz_ceres.cpp Makefile
	@echo "CXX [$(notdir $<)]"
	@g++ -Wall -O3 \
		-c $< \
		-o $(BLD_DIR)/$(basename $(notdir $<)).o \
		-I/usr/include/eigen3

$(BLD_DIR)/libglad.a:
	@gcc -c deps/glad/glad.c -o $(BLD_DIR)/glad.o \
		&& ar rcs $(BLD_DIR)/libglad.a $(BLD_DIR)/glad.o

$(BLD_DIR)/libxyz.a: $(LIBXYZ_OBJS)
	@echo "AR [libxyz.a]"
	@$(AR) $(ARFLAGS) \
		$(BLD_DIR)/libxyz.a \
		$(LIBXYZ_OBJS) \
		> /dev/null 2>&1

libxyz: setup $(BLD_DIR)/libglad.a $(BLD_DIR)/libxyz.a  ## Build libxyz

# install: ## Install libxyz
# 	mkdir -p $(PREFIX)
# 	mkdir -p $(PREFIX)/lib
# 	mkdir -p $(PREFIX)/include
# 	ln -sf $(CUR_DIR)/xyz.py $(PYTHON3_PATH)/xyz.py
# 	ln -sf $(CUR_DIR)/build/libxyz.a $(PREFIX)/lib/libxyz.a
# 	ln -sf $(CUR_DIR)/*.h $(PREFIX)/include/*.h
# 	ln -sf $(CUR_DIR)/xyz.h         $(PREFIX)/include/xyz.h
# 	ln -sf $(CUR_DIR)/xyz_ceres.h   $(PREFIX)/include/xyz_ceres.h
# 	ln -sf $(CUR_DIR)/xyz_gui.h     $(PREFIX)/include/xyz_gui.h

# uninstall: ## Uninstall libxyz
# 	rm $(PYTHON3_PATH)/xyz.py
# 	rm $(PREFIX)/lib/libxyz.a
# 	rm $(PREFIX)/include/xyz.h
# 	rm $(PREFIX)/include/xyz_ceres.h
# 	rm $(PREFIX)/include/xyz_gui.h

tests: $(TESTS) ## Build tests

run_tests: tests ## Run tests
	@cd ./build && $(foreach TEST, $(TESTS), ./$(notdir ${TEST});)

tools:
	@gcc -c tools/calib_camera.c -o $(BLD_DIR)/calib_camera

ci: ## Run CI tests
	@make tests CI_MODE=1 --no-print-directory
	@./build/test_xyz

all: libxyz tests

cppcheck: ## Run cppcheck on xyz.c
	# @cppcheck src/xyz.c src/xyz.h
	@cppcheck src/xyz_gui.c src/xyz_gui.h
