# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

llvm_canonicalize_cmake_booleans(
  CUDA_FOUND
  CUDAQ_TEST_REMOTE_SIM
)

if (CUDAQ_DISABLE_RUNTIME)
  set(CUDAQ_EXTRA_CONFIG_EXCLUDES "NVQPP")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  ${PATHS_FOR_PLUGINS}
  )

set(NVQPP_TEST_PARAMS
  nvqpp_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

get_property(test_cudaq_libraries GLOBAL PROPERTY CUDAQ_RUNTIME_LIBS)

set(NVQPP_TEST_DEPENDS
    CircuitCheck
    cudaq-opt
    cudaq-translate
    FileCheck
    test_argument_conversion
    CustomPassPlugin
)
if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  set(NVQPP_TEST_DEPENDS ${NVQPP_TEST_DEPENDS}
    cudaq-quake
    fixup-linkage
    nvq++
    ${test_cudaq_libraries}
  )
endif()

add_custom_target(nvqpp-test-depends DEPENDS ${NVQPP_TEST_DEPENDS})
set_target_properties(nvqpp-test-depends PROPERTIES FOLDER "Tests")

add_lit_testsuite(check-nvqpp "Running the nvqpp regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${NVQPP_TEST_PARAMS}
  DEPENDS ${NVQPP_TEST_DEPENDS}
)
set_target_properties(check-nvqpp PROPERTIES FOLDER "Tools")

# In case of standalone builds.
if (NVQPP_STANDALONE_BUILD)
  add_lit_testsuites(NVQPP ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS ${NVQPP_TEST_PARAMS}
    DEPENDS ${NVQPP_TEST_DEPENDS}
    FOLDER "Tests/Subdirectories"
  )
else()
  add_lit_testsuites(NVQPP ${CMAKE_CURRENT_SOURCE_DIR}
    PARAMS ${NVQPP_TEST_PARAMS}
    DEPENDS ${NVQPP_TEST_DEPENDS}
    FOLDER "Tests/Subdirectories"
  )
endif()

# Add nvqpp tests to the ctest suite
add_test(NAME ctest-nvqpp
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target check-nvqpp
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Setup an alias for 'check-all'.
add_custom_target(check)
add_dependencies(check check-all)
set_target_properties(check PROPERTIES FOLDER "Tests")
