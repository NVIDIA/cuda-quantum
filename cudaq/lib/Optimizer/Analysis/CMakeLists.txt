# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_library(OptAnalysis
  PARTIAL_SOURCES_INTENDED
  CircuitValidation.cpp
  UnitaryBuilder.cpp
  UnitaryOpGrouping.cpp

  DEPENDS
    CCDialect
    QuakeDialect
)

# UnitaryBuilder builds dense unitaries with Eigen (via runtime/common/EigenDense.h).
target_include_directories(OptAnalysis
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/eigen
  PRIVATE ${CMAKE_SOURCE_DIR}/runtime
)

register_cudaq_mlir_lib(OptAnalysis)

# The Clifford tableau oracle compiles Quake into a stabilizer tableau using the
# vendored Stim library.
add_library(OptValidationStim STATIC TableauBuilder.cpp)
add_dependencies(OptValidationStim CCDialect QuakeDialect)
target_include_directories(OptValidationStim
  PRIVATE
    ${CMAKE_SOURCE_DIR}/tpls/Stim/src
    ${CMAKE_BINARY_DIR}/cudaq/include
    ${CMAKE_SOURCE_DIR}/cudaq/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(OptValidationStim PRIVATE libstim QuakeDialect CCDialect)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  target_compile_options(OptValidationStim PRIVATE -mavx2 -msse2)
endif()
target_compile_options(OptValidationStim PRIVATE -Wno-deprecated-copy)
