# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-braced-scalar-init")
endif()

add_cudaq_library(OptCodeGen
  CCToLLVM.cpp
  CodeGenDialect.cpp
  CodeGenOps.cpp
  CodeGenTypes.cpp
  ConvertCCToLLVM.cpp
  ConvertToExecMgr.cpp
  ConvertToQIRProfile.cpp
  ConvertToQIR.cpp
  ConvertToQIRAPI.cpp
  OptUtils.cpp
  Passes.cpp
  Pipelines.cpp
  QirInsertArrayRecord.cpp
  QuakeToCodegen.cpp
  QuakeToExecMgr.cpp
  QuakeToLLVM.cpp
  RemoveMeasurements.cpp
  ReturnToOutputLog.cpp
  TranslateToIQMJson.cpp
  TranslateToOpenQASM.cpp
  VerifyNVQIRCalls.cpp
  VerifyQIRProfile.cpp
  WireSetsToProfileQIR.cpp

  DEPENDS
    CCDialect
    CodeGenDialectIncGen
    CodeGenOpsIncGen
    CodeGenTypesIncGen
    OptCodeGenPassIncGen
    OptTransformsPassIncGen
    QuakeDialect
    
  LINK_LIBS PUBLIC
    CCDialect
    OptimBuilder
    QuakeDialect

    MLIRArithTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIROpenACCDialect
    MLIROpenMPDialect
    MLIRTransforms

    # Conversions
    MLIRArithToLLVM
    MLIRComplexToLibm
    MLIRComplexToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMathToFuncs
    MLIRMathToLLVM

    # Translation
    MLIRTargetLLVMIRExport
)
