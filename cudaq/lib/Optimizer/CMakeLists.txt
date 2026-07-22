# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Register a library for inclusion in the cudaqMLIR bundle.
function(register_cudaq_mlir_lib name)
  set_property(GLOBAL APPEND PROPERTY CUDAQ_MLIR_BUNDLE_LIBS ${name})
endfunction()

add_subdirectory(Builder)
add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(CAPI)
add_subdirectory(Analysis)

# Define the shared core library libcudaqMLIR.so
include(cudaqMLIR-shlib.cmake)
