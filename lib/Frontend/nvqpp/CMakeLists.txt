# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-mlirgen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
add_mlir_library(${LIBRARY_NAME}
    # bridge from AST to Quake
    ASTBridge.cpp
    ConvertDecl.cpp
    ConvertExpr.cpp
    ConvertStmt.cpp
    ConvertType.cpp

  LINK_LIBS PUBLIC
    CCDialect
    QuakeDialect
    OptimBuilder

    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTranslateLib
    MLIRSupport
    MLIROptLib
    MLIRExecutionEngine
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    clangBasic
    clangAST
    clangFrontend
    clangFrontendTool
    clangTooling
    clangRewrite
    clangSerialization
    clangLex
)

target_compile_options(${LIBRARY_NAME} PUBLIC "-fexceptions")
