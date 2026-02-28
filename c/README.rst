Build
=====

For convenience there is a ``Makefile`` that automates the installation of
dependencies and building of ``xyz``. To install dependencies, build and test
``xyz`` run `make libxyz` and `make tests`.

Other make targets include:

.. code-block::

  all       Buld all
  deps      Install dependencies
  libxyz    Build libxyz
  tests     Build and run tests
  ci        Run CI tests
  cppcheck  Run cppcheck
  clean     Clean
  docs      Build docs

alternatively just type ``make help`` to bring up info on make targets.
