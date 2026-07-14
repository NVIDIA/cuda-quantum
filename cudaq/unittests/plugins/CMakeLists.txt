# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_file(test_external_plugin_target.sh.in
               test_external_plugin_target.sh @ONLY)
configure_file(test_install_plugin.sh.in
               test_install_plugin.sh @ONLY)

add_test(NAME external-plugin-target
  COMMAND bash test_external_plugin_target.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# The script exits 77 to signal a skip when the build did not embed a usable
# CUDA-Q version (e.g. a shallow checkout without tags), so ctest reports the
# test as skipped rather than failed. The version logic itself is covered by the
# TargetConfigTester C++ unit tests.
set_tests_properties(external-plugin-target PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME install-plugin
  COMMAND bash test_install_plugin.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
