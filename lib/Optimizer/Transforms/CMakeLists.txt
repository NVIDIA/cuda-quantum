# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
  ApplyControlNegations.cpp
  ApplyOpSpecialization.cpp
  ArgumentSynthesis.cpp
  BasisConversion.cpp
  ClassicalOptimization.cpp
  CombineMeasurements.cpp
  CombineQuantumAlloc.cpp
  ConstPropComplex.cpp
  Decomposition.cpp
  DecompositionPatterns.cpp
  DelayMeasurements.cpp
  DeleteStates.cpp
  EraseNoise.cpp
  EraseNopCalls.cpp
  ExpandControlVeqs.cpp
  ExpandMeasurements.cpp
  FactorQuantumAlloc.cpp
  GenKernelExecution.cpp
  GenDeviceCodeLoader.cpp
  GetConcreteMatrix.cpp
  GlobalizeArrayValues.cpp
  LambdaLifting.cpp
  LiftArrayAlloc.cpp
  LinearCtrlRelations.cpp
  LoopAnalysis.cpp
  LoopNormalize.cpp
  LoopPeeling.cpp
  LoopUnroll.cpp
  LowerToCFG.cpp
  LowerUnwind.cpp
  Mapping.cpp
  MemToReg.cpp
  DependencyAnalysis.cpp
  MultiControlDecomposition.cpp
  ObserveAnsatz.cpp
  PruneCtrlRelations.cpp
  PySynthCallableBlockArgs.cpp
  QuakeAddMetadata.cpp
  QuakeSynthesizer.cpp
  RefToVeqAlloc.cpp
  RegToMem.cpp
  StatePreparation.cpp
  UnitarySynthesis.cpp
  UpdateRegisterNames.cpp
  WiresToWiresets.cpp
  WriteAfterWriteElimination.cpp

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

target_include_directories(OptTransforms SYSTEM
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/eigen
  PRIVATE ${CMAKE_SOURCE_DIR}/runtime
)
