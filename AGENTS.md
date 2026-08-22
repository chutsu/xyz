# AGENTS.md

## Project Overview

`xyz` is a C99 library for robotics. It lives in `src/xyz.c` and `src/xyz.h`.

## Build Commands

- `make libxyz` - Build the library (also generates compile_commands.json if bear/compiledb is available)
- `make tests` - Build and run tests
- `make ci` - Run CI tests (sets `CI_MODE=1`)
- `make cppcheck` - Run static analysis
- `make clean` - Remove build directory
- `make help` - List all make targets

## Testing

Tests use the munit framework (`src/munit.h`). Test files live in `src/test_*.c`. Run tests with `make tests`.

## Code Style

- C99 with `-std=c99`
- Compiler: clang (configurable in `config.mk`)
- Uses `-Wall -Wpedantic -Wstrict-prototypes`
- Code formatting managed by `.clang-format`
- Python files use `.style.yapf`

## Key Paths

- `src/` - Library source and tests
- `deps/` - Third-party dependencies
- `build/` - Build output directory
- `config.mk` - Build configuration (flags, paths, targets)
- `Makefile` - Build system entry point

## Dependencies

Install with `make deps`. Includes OpenCV, OpenGL (GLFW, GLAD), SuiteSparse, BLAS/LAPACK, FreeType, Assimp, and AprilTag.
