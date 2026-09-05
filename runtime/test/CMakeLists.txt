# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Hybrid tests. Build the executable like any of the other tools, but
# run it like it was a lit test.
link_directories(${CMAKE_BINARY_DIR}/lib)

function(add_cudaq_hybrid_test TEST_NAME)
  add_llvm_executable(${TEST_NAME} Regress/${TEST_NAME}.cpp PARTIAL_SOURCES_INTENDED)

  target_compile_options(${TEST_NAME} PUBLIC -Wno-type-limits -fexceptions -DCUDAQ_RTTI_DISABLED)

  target_include_directories(${TEST_NAME}
    PUBLIC
      ${CMAKE_SOURCE_DIR}/runtime
  )

  target_link_libraries(${TEST_NAME}
    PUBLIC
      cudaq-mlir-runtime
      cudaq
      cudaq-platform-default
      cudaqMLIR)

  set_property(TARGET ${TEST_NAME} PROPERTY FOLDER test)
endfunction()

add_cudaq_hybrid_test(argument_conversion)
add_cudaq_hybrid_test(compile_target)
add_cudaq_hybrid_test(compile_target_conditional_feedback_crash)
add_cudaq_hybrid_test(resource_count_phase)
add_cudaq_hybrid_test(resource_count_qubit_indices)
add_cudaq_hybrid_test(resource_count_wireset)

# lit-based regression testing over the hybrid tests above. Each is a
# self-contained executable invoked directly by its own embedded RUN line
# (see add_cudaq_hybrid_test). This gives the runtime subsystem its own
# llvm-lit test area, separate from cudaq/test, so these tests are not
# piggybacking on the cudaq lit suite for their test infrastructure.
if (CUDAQ_BUILD_TESTS AND NOT CUDAQ_DISABLE_CPP_FRONTEND)
  if(NOT TARGET FileCheck)
    add_executable(FileCheck IMPORTED)
    set_target_properties(FileCheck PROPERTIES IMPORTED_LOCATION "/usr/local/llvm/bin/FileCheck")
  endif()

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

  set(RUNTIME_TEST_PARAMS
    cudaq_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

  set(RUNTIME_TEST_DEPENDS
    FileCheck
    argument_conversion
    compile_target
    compile_target_conditional_feedback_crash
    resource_count_phase
    resource_count_qubit_indices
    resource_count_wireset
  )

  add_custom_target(runtime-test-depends DEPENDS ${RUNTIME_TEST_DEPENDS})
  set_target_properties(runtime-test-depends PROPERTIES FOLDER "Tests")

  add_lit_testsuite(check-runtime "Running the runtime regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${RUNTIME_TEST_PARAMS}
    DEPENDS ${RUNTIME_TEST_DEPENDS}
  )
  set_target_properties(check-runtime PROPERTIES FOLDER "Tools")

  # Add runtime tests to the ctest suite.
  add_test(NAME ctest-runtime
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-runtime
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  # This test runs llvm-lit internally with parallelism, so have ctest
  # reserve all processors to prevent concurrent scheduling.
  include(ProcessorCount)
  ProcessorCount(NPROC)
  if(NPROC EQUAL 0)
    set(NPROC 1)
  endif()
  set_tests_properties(ctest-runtime PROPERTIES
    TIMEOUT 3600
    PROCESSORS ${NPROC})
endif()
