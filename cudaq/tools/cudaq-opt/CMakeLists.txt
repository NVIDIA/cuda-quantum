# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(cudaq-opt cudaq-opt.cpp)

llvm_update_compile_flags(cudaq-opt)

if(CUDAQ_MLIR_SHARED_PROVIDERS)
  # Resolve the MLIR implementation from libcudaqMLIR and the shared upstream
  # providers; do not embed a second static copy into the tool.
  target_link_libraries(cudaq-opt
    PUBLIC
    cudaq::MLIR
    CUDAQSupport
    CUDAQTargetCatalog  # The collection of pre-built targets
  )
else()
  # Link MLIR statically for performance
cudaq_use_static_mlir(cudaq-opt)
target_link_libraries(cudaq-opt
  PUBLIC
  MLIRLLVMDialect
  MLIROptLib
  MLIRIR
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRArithDialect

  CCDialect
  QECDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  CUDAQSupport
  CUDAQTargetCatalog  # The collection of pre-built targets
)
mlir_check_all_link_libraries(cudaq-opt)
endif()

export_executable_symbols_for_plugins(cudaq-opt)

install(TARGETS cudaq-opt DESTINATION bin COMPONENT Nvqpp)
