# prototype

`prototype` contain notes and code used to learn about different aspects of
robotics. From estimation, mapping to control and planning.


## Build

For convenience there is a `Makefile` that automates the installation of
dependencies and building of `prototype`, the make targets are as follows.

    debug:
      Build prototype in debug mode.

    release:
      Build prototype in release mode.

    install:
      Install prototype to $PREFIX. By default this is "/usr/local".

    deps:
      Install prototype dependencies. The dependencies are:
      - apriltags
      - boost
      - ceres
      - eigen
      - geographiclib
      - opencv3
      - realsense
      - yamlcpp

    format_code:
      Format prototype code using clang-format.

    docs:
      Generate docs for prototype.

The command one really needs to get started is `debug`, `release` and `install`.

Or, if you're old-fashioned the standard way to build a C++ project is to enter
the following commands at the root of the repo.

    mkdir -p build
    cd build
    cmake ..
    make
    sudo make install  # By default will install to /usr/local

## License

Copyright (c) <2019> <Chris Choi>. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. All advertising materials mentioning features or use of this software must
display the following acknowledgement: This product includes software developed
by Chris Choi.

4. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
