# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Add the custom target config generator test to the build, 
# it will generate the yml file in the build directory required for a lit test.
add_subdirectory(TargetConfig/custom_sim)

llvm_canonicalize_cmake_booleans(
  CUDA_FOUND
  CUDAQ_TEST_REMOTE_SIM
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
  )

set(CUDAQ_TEST_PARAMS
  nvqpp_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

get_property(test_cudaq_libraries GLOBAL PROPERTY CUDAQ_RUNTIME_LIBS)

set(CUDAQ_TEST_DEPENDS
    CircuitCheck
    cudaq-opt
    cudaq-translate
    FileCheck
)
# We require split-file, which should be installed along with FileCheck, but
# the CI doesn't do it. Comment this out and open a bug.
#   split-file
if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  set(CUDAQ_TEST_DEPENDS ${CUDAQ_TEST_DEPENDS}
    cudaq-quake
    fixup-linkage
    nvq++
    ${test_cudaq_libraries}
  )
endif()

add_custom_target(nvqpp-targettest-depends DEPENDS ${CUDAQ_TEST_DEPENDS})
set_target_properties(nvqpp-targettest-depends PROPERTIES FOLDER "TargetTests")

# Limit the number of llvm-lit worker threads because running too many of these
# tests in parallel can consume all the GPU resources.
set(LIT_ARGS "-j 8")

add_lit_testsuite(check-targets "Running the end-to-end target tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${CUDAQ_TEST_PARAMS}
  DEPENDS ${CUDAQ_TEST_DEPENDS}
  ARGS ${LIT_ARGS}
)
set_target_properties(check-targets PROPERTIES FOLDER "TargetTests")

add_lit_testsuites(CUDAQ ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${CUDAQ_TEST_PARAMS}
  DEPENDS ${CUDAQ_TEST_DEPENDS}
  FOLDER "Tests/Subdirectories"
  ARGS ${LIT_ARGS}
)

# Add nvqpp tests to the ctest suite
add_test(NAME ctest-targettests
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-targets
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
