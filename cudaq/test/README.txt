Overview of CUDA-Q Testing
--------------------------------

There are several directories in the repo related to testing. Please make sure
that tests conform to the following general hierarchy when writing tests and
reviewing PRs.

test

  The tests in this directory tree are small regression tests that can be run
  very quickly and test small combinations of functionality to prevent
  regressions. These tests use shell script command lines, various cudaq tools,
  FileCheck, etc.

unittests

  As the name implies, these are unit tests. Like the test directory, these
  tests are expected to run quickly and test small scopes of functionality.
  Unit tests may test boundary conditions, out-of-range values, etc. to
  verify that the software unit is behaving according to its specification.
  These tests make use of the gtest framework.

targettests

  The tests in this directory are end-to-end tests to test the complete path of
  functionality from source to a QPU target, which may be a specific simulator
  or hardware. These tests may take a much longer time to execute and will
  often test multiple tools in the ecosystem. They may even depend on network
  interfaces being operational, etc.

  Note: At some future point in time, the tests in targettests will be used to
  test out-of-tree on CUDA-Q installations. Such installations will not
  have the source code around. Be mindful of writing tests that use the content
  or paths of other files in other areas of the repo.

python/tests

  The python tests, for historical reasons, are broken out into their own
  directory at present. CUDA-Q support for the python language continues
  to develop and improve. At some point, CUDA-Q will integrate both C++ and
  python and these tests will be merged with and into the 3 directories explained
  above. In the meantime, python specific tests will continue to be placed in
  the python/tests directory.
