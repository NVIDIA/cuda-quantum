# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Core-only installs support native backends without the CUDA-Q compiler or
# frontend execution platform. Only targets supplied by core are imported here.
include(CMakeFindDependencyMacro)
get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(CUDAQ_INSTALL_DIR "${CUDAQ_CMAKE_DIR}/../../.." ABSOLUTE)
set(CUDAQ_LIBRARY_DIR "${CUDAQ_INSTALL_DIR}/lib")
set(CUDAQ_INCLUDE_DIR "${CUDAQ_INSTALL_DIR}/include")

# The SDK consumer discovery sequence (onsen #2, note_79022983) resolves this
# prefix's LLVM/MLIR configs: seed them so a foreign system fallback cannot
# silently substitute same-version headers or libraries.
foreach(_cudaq_cmake_root "${CUDAQ_CMAKE_DIR}/..")
  if(NOT MLIR_DIR AND EXISTS "${_cudaq_cmake_root}/mlir/MLIRConfig.cmake")
    get_filename_component(MLIR_DIR "${_cudaq_cmake_root}/mlir" ABSOLUTE)
  endif()
  if(NOT LLVM_DIR AND EXISTS "${_cudaq_cmake_root}/llvm/LLVMConfig.cmake")
    get_filename_component(LLVM_DIR "${_cudaq_cmake_root}/llvm" ABSOLUTE)
  endif()
endforeach()
unset(_cudaq_cmake_root)

# The SDK consumer discovery sequence (onsen #2, note_79022983) resolves this
# prefix's LLVM/MLIR configs: seed them so a foreign system fallback cannot
# silently substitute same-version headers or libraries.
foreach(_cudaq_cmake_root "${CUDAQ_CMAKE_DIR}/..")
  if(NOT MLIR_DIR AND EXISTS "${_cudaq_cmake_root}/mlir/MLIRConfig.cmake")
    get_filename_component(MLIR_DIR "${_cudaq_cmake_root}/mlir" ABSOLUTE)
  endif()
  if(NOT LLVM_DIR AND EXISTS "${_cudaq_cmake_root}/llvm/LLVMConfig.cmake")
    get_filename_component(LLVM_DIR "${_cudaq_cmake_root}/llvm" ABSOLUTE)
  endif()
endforeach()
unset(_cudaq_cmake_root)
find_dependency(CUDAQOperator HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQLogger HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQCommon HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(NVQIR HINTS "${CUDAQ_CMAKE_DIR}/../nvqir")
# CUDA-Q-namespaced imports of the shared upstream providers must exist
# before the core targets that reference them.
include("${CUDAQ_CMAKE_DIR}/CUDAQMLIRProviders.cmake" OPTIONAL)
include("${CUDAQ_CMAKE_DIR}/CUDAQCoreTargets.cmake")
include("${CUDAQ_CMAKE_DIR}/CUDAQPythonBindingsConfig.cmake")
