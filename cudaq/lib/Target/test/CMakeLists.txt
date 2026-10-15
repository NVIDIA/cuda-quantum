# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_program(_target_catalog_lit NAMES llvm-lit lit
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  PATHS /usr/local/llvm/bin
  NO_CACHE)
find_program(_target_catalog_filecheck NAMES FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  PATHS /usr/local/llvm/bin
  NO_CACHE)
if (NOT _target_catalog_lit OR NOT _target_catalog_filecheck)
  message(FATAL_ERROR
    "The target-catalog test suite requires llvm-lit and FileCheck, but they "
    "were not found; make them findable or reconfigure with "
    "CUDAQ_BUILD_TESTS=OFF.")
endif()

# LLVM-free driver linking the final catalog for testing.
add_executable(TargetCatalogTestDriver TargetCatalogTestDriver.cpp)
target_include_directories(TargetCatalogTestDriver PRIVATE
  ${CUDAQ_TARGET_REPO_ROOT}/cudaq/lib/Target)
target_link_libraries(TargetCatalogTestDriver
  PRIVATE CUDAQTargetCatalog)
target_compile_options(TargetCatalogTestDriver PRIVATE "-fexceptions" "-frtti")
set_target_properties(TargetCatalogTestDriver PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py @ONLY)

# Register the target-catalog test suite for CMake and CTest.
add_custom_target(check-target-catalog
  COMMAND "${_target_catalog_lit}" -sv ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS TargetCatalogTestDriver cudaq-target-db-gen
  COMMENT "Running the target-catalog LIT suite")
set_target_properties(check-target-catalog PROPERTIES FOLDER "Tests")

add_test(NAME ctest-target-catalog
  COMMAND "${_target_catalog_lit}" -sv ${CMAKE_CURRENT_BINARY_DIR})
include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC EQUAL 0)
  set(NPROC 1)
endif()
set_tests_properties(ctest-target-catalog PROPERTIES
  LABELS target-catalog
  TIMEOUT 300
  PROCESSORS ${NPROC})
