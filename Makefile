include config.mk
.PHONY: benchmarks build docs scripts src deps tools test-build test-run _libxyz_internal

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;34m%-12s\033[0m%s\n", $$1, $$2}'

setup:
	@mkdir -p $(BLD_DIR)
	@cp -r deps/fonts $(BLD_DIR)
	@cp -r src/test_data $(BLD_DIR)

$(BLD_DIR)/test_%: src/test_%.c $(BLD_DIR)/libxyz.a
	@echo "TEST [$(notdir $@)]"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lxyz

$(BLD_DIR)/%.o: src/%.c src/%.h Makefile
	@echo "CC [$(notdir $<)]"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BLD_DIR)/libglad.a:
	@gcc -c deps/src/glad/glad.c -o $(BLD_DIR)/glad.o \
		&& ar rcs $(BLD_DIR)/libglad.a $(BLD_DIR)/glad.o

$(BLD_DIR)/libxyz.a: $(LIBXYZ_OBJS)
	@echo "AR [libxyz.a]"
	@$(AR) $(ARFLAGS) \
		$(BLD_DIR)/libxyz.a \
		$(LIBXYZ_OBJS) \
		> /dev/null 2>&1

all: deps libxyz ci ## Buld all

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

venv: ## Setup env
	@python3 -m venv venv && \
	venv/bin/pip3 install -r requirements.txt && \
	echo "Run 'source venv/bin/activate' to activate the virtualenv"

compile_commands: ## Generate compile_commands.json
	@if command -v bear > /dev/null 2>&1; then \
		bear -- $(MAKE) _libxyz_internal; \
	elif command -v compiledb > /dev/null 2>&1; then \
		compiledb -n $(MAKE) _libxyz_internal; \
	else \
		echo "Error: install bear or compiledb"; exit 1; \
	fi
	@mv compile_commands.json $(BLD_DIR)/

libxyz: ## Build libxyz
	@if command -v bear > /dev/null 2>&1; then \
		bear -- $(MAKE) _libxyz_internal; \
		mv compile_commands.json $(BLD_DIR)/; \
	elif command -v compiledb > /dev/null 2>&1; then \
		compiledb -n $(MAKE) _libxyz_internal; \
		mv compile_commands.json $(BLD_DIR)/; \
	else \
		$(MAKE) _libxyz_internal; \
	fi

_libxyz_internal: \
	setup \
	$(BLD_DIR)/libglad.a \
	$(BLD_DIR)/libxyz.a \
	$(TESTS)

tests: libxyz ## Build and run tests
	@cd ./build && $(foreach TEST, $(TESTS), ./$(notdir ${TEST});)

tools:
	@gcc -c tools/calib_camera.c -o $(BLD_DIR)/calib_camera

ci: ## Run CI tests
	@make tests CI_MODE=1 --no-print-directory

cppcheck: ## Run cppcheck
	@cppcheck src/xyz.c src/xyz.h

clean:  ## Clean
	@rm -rf $(BLD_DIR)

docs: ## Build docs
	@cd docs && livereload .
