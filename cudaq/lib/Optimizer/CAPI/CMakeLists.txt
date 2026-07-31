# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_public_c_api_library(cudaqDialectsCAPI
  Dialects.cpp

  DEPENDS
  QuakeDialectIncGen
  QECOpsIncGen

  LINK_LIBS PRIVATE
  cudaqMLIR
)

# The Python C-API shims are embedded as *object files* into libcudaqMLIRCAPI
# (see python/extension/CMakeLists.txt), which links libcudaqMLIR dynamically
# and never links the static MLIR/LLVM archives itself. libcudaqMLIR must
# therefore export every library those shims were built against.
#
# This has to be recorded here rather than next to the library that embeds them:
# libcudaqMLIR is created at the end of cudaq/lib/Optimizer/CMakeLists.txt,
# long before add_subdirectory(python), so CUDAQ_MLIR_REQUIRED_LIBS is already
# frozen by the time python/ is configured.
set(CUDAQ_PYTHON_CAPI_LIBS
  cudaqDialectsCAPI
  MLIRCAPIDebug
  MLIRCAPIExecutionEngine
  MLIRCAPIFunc
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILLVM
  MLIRCAPITarget
  MLIRCAPITransforms
  CACHE INTERNAL "C-API libraries embedded into libcudaqMLIRCAPI")

foreach(_capi_lib IN LISTS CUDAQ_PYTHON_CAPI_LIBS)
  get_target_property(_capi_deps ${_capi_lib} MLIR_AGGREGATE_DEP_LIBS_IMPORTED)
  foreach(_dep IN LISTS _capi_deps)
    if(TARGET ${_dep}
        AND NOT _dep IN_LIST CUDAQ_PYTHON_CAPI_LIBS
        AND NOT _dep MATCHES "CAPI"
        AND NOT _dep STREQUAL "cudaqMLIR")
      set_property(GLOBAL APPEND PROPERTY CUDAQ_MLIR_REQUIRED_LIBS "${_dep}")
    endif()
  endforeach()
endforeach()
