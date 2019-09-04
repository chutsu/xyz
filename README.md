# proto [![Build Status](https://github.com/chutsu/proto/workflows/C/C++%20CI/badge.svg)][1]

`proto` contain notes and code used to learn about different aspects of
robotics. From estimation, mapping to control and planning.


## Build

For convenience there is a `Makefile` that automates the installation of
dependencies and building of `proto`, the make targets are as follows.

    deps:
      Install proto dependencies.

    debug:
      Build proto in debug mode.

    release:
      Build proto in release mode.

    install:
      Install proto to $PREFIX. By default this is "/usr/local".

    format_code:
      Format proto code using clang-format.

    docs:
      Generate docs for proto.

Or, the standard way to build a C++ project is to enter the following commands
at the root of the repo.

    mkdir -p build
    cd build
    cmake ..
    make
    sudo make install  # By default will install to /usr/local

## License

The source code is released under GPLv3 license.

[1]: https://github.com/chutsu/proto/actions
