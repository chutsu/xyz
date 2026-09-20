include config.mk

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile \
		| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[1;34m%-12s\033[0m%s\n", $$1, $$2}'

################################################################################
# BUILD RULES
################################################################################

# Make only tracks file mtimes, not variable content, so a flags-only
# change (e.g. `make ci` setting CI_MODE=1) would otherwise leave an
# already-up-to-date .o/binary stale and unrebuilt. These sentinel
# files record the flags used last build; FORCE makes them re-checked
# every run, but they only touch (and so only trigger a rebuild) when
# the flags actually changed.
$(BLD_DIR)/.cflags: FORCE
	@mkdir -p $(BLD_DIR)
	@echo "$(CFLAGS)" > $(BLD_DIR)/.cflags.tmp
	@cmp -s $(BLD_DIR)/.cflags.tmp $@ 2>/dev/null \
		|| mv $(BLD_DIR)/.cflags.tmp $@
	@rm -f $(BLD_DIR)/.cflags.tmp

$(BLD_DIR)/.cxxflags: FORCE
	@mkdir -p $(BLD_DIR)
	@echo "$(CXXFLAGS)" > $(BLD_DIR)/.cxxflags.tmp
	@cmp -s $(BLD_DIR)/.cxxflags.tmp $@ 2>/dev/null \
		|| mv $(BLD_DIR)/.cxxflags.tmp $@
	@rm -f $(BLD_DIR)/.cxxflags.tmp

.PHONY: FORCE
FORCE:

$(BLD_DIR)/test_%: src/test_%.c $(BLD_DIR)/libxyz.a $(BLD_DIR)/.cflags
	@echo "TEST [$(notdir $@)]"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) -lxyz

$(BLD_DIR)/benchmark_%: \
		src/benchmark_%.cpp \
		$(BLD_DIR)/libxyz.a \
		$(BLD_DIR)/.cxxflags
	@echo "BENCHMARK [$(notdir $@)]"
	@$(CXX) $(CXXFLAGS) $< -o $@ $(CXXLDFLAGS) -lxyz

$(BLD_DIR)/%.o: src/%.c src/%.h Makefile $(BLD_DIR)/.cflags
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

################################################################################
# TARGETS
################################################################################

.PHONY: all
all: deps libxyz ci ## Buld all

.PHONY: deps
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

.PHONY: venv
venv: ## Setup env
	@python3 -m venv venv && \
	venv/bin/pip3 install -r requirements.txt && \
	echo "Run 'source venv/bin/activate' to activate the virtualenv"

.PHONY: setup
setup:
	@mkdir -p $(BLD_DIR)
	@cp -r deps/fonts $(BLD_DIR)
	@cp -r src/test_data $(BLD_DIR)

.PHONY: libxyz
libxyz: ## Build libxyz
	@if command -v bear > /dev/null 2>&1; then \
		bear -- $(MAKE) -s _libxyz_internal; \
		mv compile_commands.json $(BLD_DIR)/; \
	elif command -v compiledb > /dev/null 2>&1; then \
		compiledb -n $(MAKE) -s _libxyz_internal; \
		mv compile_commands.json $(BLD_DIR)/; \
	else \
		$(MAKE) -s _libxyz_internal; \
	fi

.PHONY: _libxyz_internal
_libxyz_internal: \
	setup \
	$(BLD_DIR)/libglad.a \
	$(BLD_DIR)/libxyz.a \
	$(TESTS)

.PHONY: tests
tests: libxyz ## Build and run tests
	@cd ./build && $(foreach TEST, $(TESTS), ./$(notdir ${TEST});)

# Benchmarks are meaningless under ASan/debug, so force a release libxyz
# regardless of the ambient BUILD_TYPE (this leaves build/libxyz.a in release
# form afterwards -- rerun `make libxyz` to restore the default debug build).
.PHONY: benchmark
benchmark: ## Build and run benchmarks
	@rm -f $(BLD_DIR)/xyz.o $(BLD_DIR)/libxyz.a
	@$(MAKE) -s _libxyz_internal BUILD_TYPE=release --no-print-directory
	@$(MAKE) -s $(BENCHMARKS) BUILD_TYPE=release --no-print-directory
	@cd ./build && $(foreach BENCH, $(BENCHMARKS), ./$(notdir ${BENCH});)

.PHONY: ci
ci: ## Run CI tests
	@make tests CI_MODE=1 --no-print-directory

.PHONY: cppcheck
cppcheck: ## Run cppcheck
	@cppcheck src/xyz.c src/xyz.h

.PHONY: clean
clean:  ## Clean
	@rm -rf $(BLD_DIR)

.PHONY: docs
docs: ## Build docs
	@cd docs && livereload .
