# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# qlx-lsp-server: mlir-lsp-server with the retained P0-P3 dialects loaded.
# Point the VS Code MLIR extension at this binary via
#   "mlir.server_path": "${workspaceFolder}/build/bin/qlx-lsp-server"
# in .vscode/settings.json.

add_llvm_executable(qlx-lsp-server
  qlx-lsp-server.cpp
)

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRLspServerLib
)
if(QLX_USE_CUDAQ_SDK)
  # The CUDA-Q aggregate exports the MLIR surface used by QLX dialects but not
  # the standalone LSP server entry point.
  list(APPEND _qlx_mlir_link_libraries MLIRLspServerLib)
endif()

target_link_libraries(qlx-lsp-server PRIVATE
  MLIRQLXDialect
  MLIRLVMDialect
  MLIRPhysDialect
  MLIRCflowDialect
  MLIREventDialect
  MLIRFabricDialect

  ${_qlx_mlir_link_libraries}
)

llvm_update_compile_flags(qlx-lsp-server)

set_target_properties(qlx-lsp-server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${QLX_TOOLS_DIR}
)
