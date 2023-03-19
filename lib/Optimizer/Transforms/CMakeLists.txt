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
  CCMem2Reg.cpp
  ExpandMeasurements.cpp
  FuncToQTX.cpp
  GenKernelExecution.cpp
  GenDeviceCodeLoader.cpp
  LambdaLifting.cpp
  LoopUnroll.cpp
  LowerToCFG.cpp
  LowerUnwind.cpp
  OpCancellation.cpp
  OpDecomposition.cpp
  Passes.cpp
  QTXToQuake.cpp
  QuakeAddMetadata.cpp
  QuakeObserveAnsatz.cpp
  QuakeSynthesizer.cpp
  QuakeToQTX.cpp
  QuakeToQTXConverter.cpp
  RaiseToAffine.cpp
  SplitArrays.cpp

  DEPENDS
    OptTransformsPassIncGen
    CCDialect
    QTXDialect
    QuakeDialect
    
  LINK_LIBS PUBLIC
    CCDialect
    MLIRIR
    OptimBuilder
    QTXDialect
    QuakeDialect
)
