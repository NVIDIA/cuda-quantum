# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-braced-scalar-init")
endif()

add_cudaq_library(OptCodeGen
  CodeGenDialect.cpp
  CodeGenOps.cpp
  CodeGenTypes.cpp
  LowerToQIRProfile.cpp
  LowerToQIR.cpp
  Passes.cpp
  RemoveMeasurements.cpp
  TranslateToIQMJson.cpp
  TranslateToOpenQASM.cpp

  DEPENDS
    CCDialect
    CodeGenDialectIncGen
    CodeGenOpsIncGen
    CodeGenTypesIncGen
    OptCodeGenPassIncGen
    OptPeepholeIncGen
    OptTransformsPassIncGen
    QuakeDialect
    
  LINK_LIBS PUBLIC
    CCDialect
    MLIRIR
    OptimBuilder
    QuakeDialect
)
