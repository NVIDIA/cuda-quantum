# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# An explicitly requested lit (LLVM_EXTERNAL_LIT, as used elsewhere in the
# build) takes precedence over searching for one.
if (LLVM_EXTERNAL_LIT)
  set(_target_catalog_lit "${LLVM_EXTERNAL_LIT}")
  if (NOT EXISTS "${_target_catalog_lit}")
    message(FATAL_ERROR
      "LLVM_EXTERNAL_LIT is set to '${LLVM_EXTERNAL_LIT}', which does not "
      "exist; the target-catalog test suite needs it. Fix the path or "
      "reconfigure with CUDAQ_BUILD_TESTS=OFF.")
  endif()
else()
  find_program(_target_catalog_lit NAMES llvm-lit lit
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    PATHS /usr/local/llvm/bin
    NO_CACHE)
  if (NOT _target_catalog_lit)
    message(FATAL_ERROR
      "The target-catalog test suite requires llvm-lit, but it was not "
      "found; set LLVM_EXTERNAL_LIT, make it findable, or reconfigure with "
      "CUDAQ_BUILD_TESTS=OFF.")
  endif()
endif()
find_program(_target_catalog_filecheck NAMES FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  PATHS /usr/local/llvm/bin
  NO_CACHE)
if (NOT _target_catalog_filecheck)
  message(FATAL_ERROR
    "The target-catalog test suite requires FileCheck, but it was not "
    "found; make it findable or reconfigure with CUDAQ_BUILD_TESTS=OFF.")
endif()

# LLVM-free driver linking the final catalog for testing.
add_executable(TargetCatalogTestDriver TargetCatalogTestDriver.cpp)
target_include_directories(TargetCatalogTestDriver PRIVATE
  ${CUDAQ_TARGET_REPO_ROOT}/cudaq/lib/Target)
# The catalog already contains reflect-cpp's objects, so take only its headers,
# as system headers like linking it would.
target_include_directories(TargetCatalogTestDriver SYSTEM PRIVATE
  $<TARGET_PROPERTY:reflectcpp,INTERFACE_INCLUDE_DIRECTORIES>)
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
