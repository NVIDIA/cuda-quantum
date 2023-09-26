# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-braced-scalar-init")
endif()

add_cudaq_library(OptTransforms
  AddDeallocs.cpp
  AggressiveEarlyInlining.cpp
  ApplyOpSpecialization.cpp
  BasisConversion.cpp
  CombineQuantumAlloc.cpp
  Decomposition.cpp
  DecompositionPatterns.cpp
  ExpandMeasurements.cpp
  FactorQuantumAlloc.cpp
  GenKernelExecution.cpp
  GenDeviceCodeLoader.cpp
  LambdaLifting.cpp
  LoopAnalysis.cpp
  LoopNormalize.cpp
  LoopUnroll.cpp
  LowerToCFG.cpp
  LowerUnwind.cpp
  MemToReg.cpp
  MultiControlDecomposition.cpp
  PruneCtrlRelations.cpp
  QuakeAddMetadata.cpp
  QuakeObserveAnsatz.cpp
  QuakeSynthesizer.cpp
  RegToMem.cpp

  DEPENDS
    OptTransformsPassIncGen
    CCDialect
    QuakeDialect
    
  LINK_LIBS PUBLIC
    CCDialect
    MLIRIR
    OptimBuilder
    QuakeDialect
)
