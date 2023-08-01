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

find_package(METIS)
if (METIS_FOUND)
  message(STATUS "Metis found, building QuakeCircuitCutPass.")

  add_cudaq_library(OptCodeGen
    LowerToBaseProfileQIR.cpp
    LowerToQIR.cpp
    Passes.cpp
    RefToVeqAlloc.cpp
    QuakeCircuitCutPass.cpp

    DEPENDS
      CCDialect
      OptCodeGenPassIncGen
      OptPeepholeIncGen
      OptTransformsPassIncGen
      CUDAQTargetOpenQASM
      QuakeDialect
      
    LINK_LIBS PUBLIC
      CCDialect
      MLIRIR
      OptimBuilder
      QuakeDialect
  )

  target_include_directories(OptCodeGen PRIVATE ${METIS_INCLUDE_DIRS})
  target_link_libraries(OptCodeGen PRIVATE ${METIS_LIBRARIES} ${METIS_LIBRARY_DIRS}/libGKlib.a)
else()
  add_cudaq_library(OptCodeGen
    LowerToBaseProfileQIR.cpp
    LowerToQIR.cpp
    Passes.cpp
    RefToVeqAlloc.cpp

    DEPENDS
      CCDialect
      OptCodeGenPassIncGen
      OptPeepholeIncGen
      OptTransformsPassIncGen
      CUDAQTargetOpenQASM
      QuakeDialect
      
    LINK_LIBS PUBLIC
      CCDialect
      MLIRIR
      OptimBuilder
      QuakeDialect
  )
endif()

