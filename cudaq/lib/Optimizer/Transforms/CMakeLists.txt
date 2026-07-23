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

add_cudaq_library(OptTransforms
  AddDeallocs.cpp
  AddMeasurements.cpp
  AddMetadata.cpp
  AggressiveInlining.cpp
  ApplyOpSpecialization.cpp
  ArgumentSynthesis.cpp
  BasisConversion.cpp
  CableRoughIn.cpp
  ClassicalOptimization.cpp
  CombineMeasurements.cpp
  CombineQuantumAlloc.cpp
  ConstantPropagation.cpp
  DeadQuantumElimination.cpp
  DeadStoreRemoval.cpp
  Decomposition.cpp
  DecompositionPatterns.cpp
  DecompositionPatternSelection.cpp
  DesugarCableOps.cpp
  DistributedDeviceCall.cpp
  EraseNoise.cpp
  EraseNopCalls.cpp
  EraseQEC.cpp
  EraseVectorCopyCtor.cpp
  ExpandControlNegations.cpp
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
  PhaseFolding.cpp
  Pipelines.cpp
  PruneCtrlRelations.cpp
  PruneNegligibleRotations.cpp
  PyFrontendCleanup.cpp
  PySynthCallableBlockArgs.cpp
  QuakePropagateMetadata.cpp
  QuakeSimplify.cpp
  QuakeSynthesizer.cpp
  RealtimeDeviceCall.cpp
  RefToVeqAlloc.cpp
  RegToMem.cpp
  RepairLinearType.cpp
  ReplaceStateWithKernel.cpp
  ResetBeforeReuse.cpp
  ResourceCountPreprocess.cpp
  RunSemanticsHackery.cpp
  SROA.cpp
  StackFramePrealloc.cpp
  StatePreparation.cpp
  UnitarySynthesis.cpp
  UpdateRegisterNames.cpp
  VariableCoalesce.cpp
  WiresToWiresets.cpp
  WriteAfterWriteElimination.cpp

  DEPENDS
    OptTransformsPassIncGen
    CCDialect
    QECDialect
    QuakeDialect
    
  LINK_LIBS PUBLIC
    CCDialect
    MLIRIR
    OptimBuilder
    QECDialect
    QuakeDialect
)

target_include_directories(OptTransforms SYSTEM
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/eigen
  PRIVATE ${CMAKE_SOURCE_DIR}/runtime
)

register_cudaq_mlir_lib(OptTransforms)
