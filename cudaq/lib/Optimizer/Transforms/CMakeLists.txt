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
  ApplyControlNegations.cpp
  ApplyOpSpecialization.cpp
  ArgumentSynthesis.cpp
  BasisConversion.cpp
  CableRoughIn.cpp
  ClassicalOptimization.cpp
  CliffordTSynthesis.cpp
  CombineMeasurements.cpp
  CombineQuantumAlloc.cpp
  ConstantPropagation.cpp
  DeadStoreRemoval.cpp
  Decomposition.cpp
  DecompositionPatterns.cpp
  DecompositionPatternSelection.cpp
  DesugarCableOps.cpp
  DistributedDeviceCall.cpp
  EraseNoise.cpp
  EraseNopCalls.cpp
  EraseVectorCopyCtor.cpp
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
  PySynthCallableBlockArgs.cpp
  QuakePropagateMetadata.cpp
  QuakeSimplify.cpp
  QuakeSynthesizer.cpp
  RefToVeqAlloc.cpp
  RegToMem.cpp
  ReplaceStateWithKernel.cpp
  ResetBeforeReuse.cpp
  ResourceCount.cpp
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

if(TARGET cudaq-synth)
  target_link_libraries(OptTransforms PRIVATE cudaq-synth)
  target_compile_definitions(obj.OptTransforms PRIVATE
    CUDAQ_HAS_CLIFFORD_T_SYNTHESIS)
  target_include_directories(obj.OptTransforms PRIVATE
    $<TARGET_PROPERTY:cudaq-synth,INTERFACE_INCLUDE_DIRECTORIES>)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(CliffordTSynthesis.cpp PROPERTIES
      COMPILE_OPTIONS "-Wno-maybe-uninitialized")
  endif()
endif()
