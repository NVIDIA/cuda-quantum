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
configure_file(test_build_target_from_config.sh.in
               test_build_target_from_config.sh @ONLY)

add_test(NAME external-plugin-target
  COMMAND bash test_external_plugin_target.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME install-plugin
  COMMAND bash test_install_plugin.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME build-target-from-config
  COMMAND bash test_build_target_from_config.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
