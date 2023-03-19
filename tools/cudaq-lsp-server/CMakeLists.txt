# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(cudaq_dialect_libs GLOBAL PROPERTY CUDAQ_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
)

set(LIBS
  ${mlir_dialect_libs}
  ${cudaq_dialect_libs}
  ${conversion_libs}
  ${test_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)

add_llvm_executable(cudaq-lsp-server
  cudaq-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cudaq-lsp-server 
  PRIVATE 
  ${LIBS}
)

llvm_update_compile_flags(cudaq-lsp-server)

mlir_check_all_link_libraries(cudaq-lsp-server)

install(TARGETS cudaq-lsp-server DESTINATION bin)
