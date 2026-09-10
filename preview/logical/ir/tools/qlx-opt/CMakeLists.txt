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

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIROptLib
  MLIRIR
  MLIRSupport
  MLIRParser
)

target_link_libraries(qlx-opt PRIVATE
  MLIRQLXDialect
  MLIRLVMDialect
  MLIRPhysDialect
  MLIRPhysTransforms
  MLIRCflowDialect
  MLIREventDialect
  MLIRQLXTransforms
  MLIRQuakeToQLX
  MLIRQLXToLVM
  MLIRFabricDialect
  MLIRFabricTransforms

  ${_qlx_mlir_link_libraries}
)

llvm_update_compile_flags(qlx-opt)

# Export all symbols so that dlopen'd pass/dialect plugins can resolve MLIR
# symbols from this binary (equivalent to -rdynamic on Linux).
set_target_properties(qlx-opt PROPERTIES
  ENABLE_EXPORTS ON
  RUNTIME_OUTPUT_DIRECTORY ${QLX_TOOLS_DIR}
)

if(QLX_USE_CUDAQ_SDK)
  qlx_select_mlir_link_libraries(_qlx_quake_mlir_link_libraries
    MLIRComplexDialect
    MLIRLoopLikeInterface
  )
  target_compile_definitions(qlx-opt PRIVATE QLX_HAS_CUDAQ_QUAKE=1)
  target_link_libraries(qlx-opt PRIVATE
    QLXCUDAQuantumDialects
    ${_qlx_quake_mlir_link_libraries}
  )
endif()

if(SKBUILD)
  set_target_properties(qlx-opt PROPERTIES
    INSTALL_RPATH "${QLX_TOOLS_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

install(TARGETS qlx-opt
  RUNTIME DESTINATION "${QLX_TOOLS_INSTALL_DIR}"
  COMPONENT QLXTools)
