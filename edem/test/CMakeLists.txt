# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(consumer)
add_executable(edem_composition_test composition.c)
target_link_libraries(edem_composition_test PRIVATE cudaq::edem)
add_test(NAME edem_composition COMMAND edem_composition_test)
add_executable(edem_core_test core.c)
target_link_libraries(edem_core_test PRIVATE cudaq::edem)
if(UNIX AND NOT APPLE)
  target_link_libraries(edem_core_test PRIVATE m)
endif()
add_test(NAME edem_core COMMAND edem_core_test)
add_executable(edem_owned_memory_test owned_memory.c)
target_link_libraries(edem_owned_memory_test PRIVATE cudaq::edem)
add_test(NAME edem_owned_memory COMMAND edem_owned_memory_test)
add_executable(edem_scaling_test scaling.c)
target_link_libraries(edem_scaling_test PRIVATE cudaq::edem)
add_test(NAME edem_scaling COMMAND edem_scaling_test)
set_tests_properties(edem_scaling PROPERTIES TIMEOUT 30)
if(CMAKE_NM)
  add_test(NAME edem_allocator_and_exports
    COMMAND "${CMAKE_COMMAND}" "-DNM=${CMAKE_NM}"
      "-DOTOOL=${CMAKE_OTOOL}" "-DREADELF=${CMAKE_READELF}"
      "-DSHARED=${BUILD_SHARED_LIBS}"
      "-DEDEM_LIBRARY=$<TARGET_FILE:cudaq-edem>"
      "-DGF2_LIBRARY=$<TARGET_FILE:cudaq-gf2>"
      "-DEDEM_EXPORT_MAP=$<TARGET_PROPERTY:cudaq-edem,CUDAQ_EXPORT_MAP>"
      "-DGF2_EXPORT_MAP=$<TARGET_PROPERTY:cudaq-gf2,CUDAQ_EXPORT_MAP>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/check_allocators.cmake")
endif()
configure_file(check_package.cmake.in check_package.cmake @ONLY)

add_test(NAME edem_installed_package
  COMMAND "${CMAKE_COMMAND}" "-DEDEM_TEST_CONFIG=$<CONFIG>"
    -P "${CMAKE_CURRENT_BINARY_DIR}/check_package.cmake")
add_test(NAME edem_parent_build
  COMMAND "${CMAKE_COMMAND}" "-DEDEM_TEST_CONFIG=$<CONFIG>"
    -DEDEM_TEST_PARENT=ON
    -P "${CMAKE_CURRENT_BINARY_DIR}/check_package.cmake")
add_test(NAME edem_custom_install_dirs
  COMMAND "${CMAKE_COMMAND}" "-DEDEM_TEST_CONFIG=$<CONFIG>"
    -DEDEM_TEST_CUSTOM_INSTALL_DIRS=ON
    -P "${CMAKE_CURRENT_BINARY_DIR}/check_package.cmake")

if(CUDAQ_EDEM_BUILD_PYTHON)
  add_test(NAME edem_python_import
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=$<TARGET_FILE_DIR:cudaq-edem-python>/.."
      "${CUDAQ_EDEM_PYTHON_EXECUTABLE}" -c "import cudaq_edem")
  add_test(NAME edem_python_values
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=$<TARGET_FILE_DIR:cudaq-edem-python>/.."
      "${CUDAQ_EDEM_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_values.py"
      --require-adoption)
endif()
