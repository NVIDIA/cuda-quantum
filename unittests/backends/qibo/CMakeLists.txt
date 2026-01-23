add_executable(test_qibo QiboTester.cpp)
target_include_directories(test_qibo PRIVATE ../..)
target_link_libraries(test_qibo
  PRIVATE fmt::fmt-header-only 
  PRIVATE
  cudaq-common
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  cudaq-platform-default 
  cudaq
  gtest_main
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QiboStartServerAndTest.sh.in
              ${CMAKE_CURRENT_BINARY_DIR}/QiboStartServerAndTest.sh @ONLY)

add_test(NAME QiboTester COMMAND ${CMAKE_CURRENT_BINARY_DIR}/QiboStartServerAndTest.sh)
set_tests_properties(QiboTester PROPERTIES TIMEOUT 120)