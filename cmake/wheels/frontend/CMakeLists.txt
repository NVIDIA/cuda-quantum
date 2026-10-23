# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.30)
# No compilation in this pass: reuse the core packaging pass's native build.
project(CUDAQFrontendWheel LANGUAGES NONE)
set(CUDAQ_WHEEL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../_skbuild"
  CACHE PATH "Native build shared by the core and frontend wheels")
if(NOT EXISTS "${CUDAQ_WHEEL_BUILD_DIR}/cmake_install.cmake")
  message(FATAL_ERROR "Build the core wheel before packaging the frontend.")
endif()
set(CUDAQ_WHEEL_KIND frontend)
configure_file(../InstallWheel.cmake.in InstallFrontendWheel.cmake @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallFrontendWheel.cmake")
