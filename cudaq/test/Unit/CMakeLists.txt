# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# this CMakeLists.txt file is meant to mimic `llvm/test/Unit/CMakeLists.txt`

# create a custom target `check-cudaq-unit` that aggregates all of the unit
# tests that we have created with the `add_cudaq_unittest` function.
get_property(CUDAQ_UNIT_TEST_TARGETS GLOBAL PROPERTY CUDAQ_UNIT_TEST_TARGETS)

set(cudaq_unit_test_commands "")
foreach(test_target IN LISTS CUDAQ_UNIT_TEST_TARGETS)
    list(APPEND cudaq_unit_test_commands
        COMMAND $<TARGET_FILE:${test_target}>
    )
endforeach()

add_custom_target(check-cudaq-unit
    ${cudaq_unit_test_commands}
    DEPENDS ${CUDAQ_UNIT_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
)

# add a CTest test that runs the `check-cudaq-unit` target when we run `ctest -R ctest-cudaq-unit`.
add_test(NAME ctest-cudaq-unit
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-cudaq-unit
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
