# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
)

set(PYCUDAQMLIR_TEST_PARAMS
  pycudaqmlir_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(PYCUDAQMLIR_TEST_DEPENDS
  CUDAQuantumPythonModules
  FileCheck)

add_custom_target(pycudaqmlir-test-depends DEPENDS ${PYCUDAQMLIR_TEST_DEPENDS})
# set_target_properties(pycudaqmlir-test-depends PROPERTIES FOLDER "Tools")
# set_target_properties(pycudaqmlir-test-depends PROPERTIES FOLDER "Tests")

add_lit_testsuite(check-pycudaq-mlir "Running cudaq python regression tests."
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${PYCUDAQMLIR_TEST_PARAMS}
  DEPENDS ${PYCUDAQMLIR_TEST_DEPENDS}
)
set_target_properties(check-pycudaq-mlir PROPERTIES FOLDER "Tests")
set_target_properties(check-pycudaq-mlir PROPERTIES FOLDER "Tools")

add_lit_testsuites(PYCUDAQMLIR ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${PYCUDAQMLIR_TEST_PARAMS}
  DEPENDS ${PYCUDAQMLIR_TEST_DEPENDS})

# Add nvqpp tests to the ctest suite
add_test(NAME pycudaq-mlir
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-pycudaq-mlir
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
