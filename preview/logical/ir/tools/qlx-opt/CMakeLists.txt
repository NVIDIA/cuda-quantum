# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# qlx-opt: mlir-opt with QLX dialect loaded.

add_llvm_executable(qlx-opt
  qlx-opt.cpp
)

target_link_libraries(qlx-opt PRIVATE
  MLIRQLXDialect
  MLIRLVMDialect
  MLIRQLXTransforms
  MLIRQuakeToQLX
  MLIRQLXToLVM
  MLIRFabricDialect
  MLIRFabricTransforms

  # CUDA-Q's Quake / CC dialects share CUDA-Q's MLIR image.
  QLXCUDAQuantumDialects

  MLIRArithDialect
  MLIRComplexDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRSCFDialect
  MLIROptLib
  MLIRIR
  MLIRSupport
  MLIRParser
)

llvm_update_compile_flags(qlx-opt)

# Link MLIR statically, like cudaq-opt.
cudaq_use_static_mlir(qlx-opt)

# Export all symbols so that dlopen'd pass/dialect plugins can resolve MLIR
# symbols from this binary (equivalent to -rdynamic on Linux).
set_target_properties(qlx-opt PROPERTIES
  ENABLE_EXPORTS ON
  RUNTIME_OUTPUT_DIRECTORY ${QLX_TOOLS_DIR}
)

if(SKBUILD)
  set_target_properties(qlx-opt PROPERTIES
    INSTALL_RPATH "${QLX_TOOLS_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

install(TARGETS qlx-opt
  RUNTIME DESTINATION "${QLX_TOOLS_INSTALL_DIR}"
  COMPONENT QLXTools)
