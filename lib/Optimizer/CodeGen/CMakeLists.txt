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

add_cudaq_library(OptCodeGen
  LowerToBaseProfileQIR.cpp
  LowerToQIR.cpp
  Passes.cpp
  RefToVeqAlloc.cpp
  TranslateToIQMJson.cpp
  TranslateToOpenQASM.cpp

  DEPENDS
    CCDialect
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
