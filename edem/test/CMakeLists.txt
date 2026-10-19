# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(consumer)
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
endif()
