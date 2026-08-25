# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check that a library is listed in `CUDAQ_MLIR_BUNDLED_LIBS`.
function(check_registered_mlir_lib name)
  if(NOT name IN_LIST CUDAQ_MLIR_BUNDLED_LIBS)
    message(FATAL_ERROR
      "${name} is not listed in ${CUDAQ_MLIR_BUNDLED_LIBS_PATH}.")
  endif()
endfunction()

add_subdirectory(Builder)
add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(CAPI)
add_subdirectory(Analysis)

# Define the shared core library libcudaqMLIR.so
include(cudaqMLIR-shlib.cmake)

# Define a static variant of the bundled MLIR library
add_library(cudaq-mlir-static INTERFACE)
target_link_libraries(cudaq-mlir-static INTERFACE ${CUDAQ_MLIR_BUNDLED_LIBS})
