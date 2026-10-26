# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Bundle LLVM/MLIR binaries and source code into the CUDA-Q install tree.
# Used when building the cudaq-devel wheel.
if(NOT CUDAQ_BUNDLE_MLIR_INSTALL)
  return()
endif()

# LLVM_DIR points at .../lib/cmake/llvm; walk up to the install prefix.
get_filename_component(CUDAQ_LLVM_INSTALL_PREFIX "${LLVM_DIR}/../../.." ABSOLUTE)

message(STATUS "Bundling LLVM/MLIR devel tree from ${CUDAQ_LLVM_INSTALL_PREFIX}")

# Subtrees copied verbatim from the upstream LLVM/MLIR install prefix.
set(_cudaq_mlir_devel_dirs
  include
  lib
  bin
  src/python
)
foreach(_dir IN LISTS _cudaq_mlir_devel_dirs)
  if(EXISTS "${CUDAQ_LLVM_INSTALL_PREFIX}/${_dir}")
    get_filename_component(_install_destination "${_dir}" DIRECTORY)
    if(NOT _install_destination)
      set(_install_destination ".")
    endif()
    install(DIRECTORY "${CUDAQ_LLVM_INSTALL_PREFIX}/${_dir}"
      DESTINATION "${_install_destination}"
      COMPONENT Development
      USE_SOURCE_PERMISSIONS)
  endif()
endforeach()

# Make the bundled LLVM tree relocatable. The rewriting runs at install time and
# lives in its own script so it reads as ordinary CMake instead of a quoted
# install(CODE) string with three levels of escaping.
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/RelocateBundledLLVM.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/RelocateBundledLLVM.cmake"
  @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/RelocateBundledLLVM.cmake"
  COMPONENT Development)
