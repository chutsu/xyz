xyz
===

``xyz`` is a C99 library for robotics. It provides data structures, linear
algebra, 3D transforms, computer vision, state estimation, sensor data
loaders, and rendering.

Modules
-------

- SYSTEM - Stack traces and terminal helpers
- DATA - Strings, file I/O, CSV/DSV parsing
- TIME - Timestamps and timing utilities
- ARRAY - Generic dynamic pointer array
- DARRAY - Typed dynamic array with expand/contract
- LIST - Doubly-linked list
- RED-BLACK-TREE - Self-balancing BST (keys and values)
- HASHMAP - Open-addressing hash map
- NETWORK - TCP server and client
- MATH - Scalar math, comparison, statistics
- LINEAR ALGEBRA - Matrix/vector ops, SVD, Cholesky, QR, eigen
- SUITE-SPARSE - CHOLMOD sparse linear algebra
- TRANSFORMS - 3D rigid transforms, rotations, quaternions
- LIE - SO(3) and S2 Lie group operations
- GNUPLOT - Gnuplot pipe interface
- CONTROL - PID controller
- MAV - Quadrotor model, controllers, waypoints
- COMPUTER-VISION - Images, camera models, projective geometry
- APRILGRID - AprilTag grid detection and layout
- MORTON CODES - 2D/3D spatial encoding
- PLANE - 3D plane representation
- FRUSTUM - View frustum and culling
- POINT CLOUD - Umeyama point cloud alignment
- VOXEL - Voxel grid and downsampling
- OCTREE - Octree spatial partitioning

Build
-----

For convenience there is a ``Makefile`` that automates the installation of
dependencies and building of ``xyz``. To install dependencies, build and test
``xyz`` run ``make libxyz`` and ``make tests``.

Other make targets include:

.. code-block::

  all        Build all
  deps       Install dependencies
  libxyz     Build libxyz
  tests      Build and run tests
  ci         Run CI tests
  cppcheck   Run cppcheck
  clean      Clean
  docs       Build docs
  venv       Setup Python virtual environment
  compile_commands  Generate compile_commands.json

Alternatively just type ``make help`` to bring up info on make targets.


License
-------

.. code-block::

  Copyright (c) <2020> <Chris Choi>

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
