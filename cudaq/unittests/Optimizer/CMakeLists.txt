# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_unittest("OptimizerUnitTests" "optimizer-unit-tests"
    SOURCES
        HermitianTrait.cpp
        FactoryMergeModuleTest.cpp
        DecompositionPatternsTest.cpp
        DecompositionPatternSelectionTest.cpp
    INCLUDES
        ${CMAKE_SOURCE_DIR}/runtime
        ${CMAKE_SOURCE_DIR}/cudaq/lib/Optimizer/Transforms
    LINKED_LIBS
        MLIRArithDialect
        MLIRFuncDialect
        MLIRParser
        MLIRPass
        MLIRTransforms
        QuakeDialect
        CCDialect
        OptTransforms
        OptimBuilder
    COMPILE_OPTS
      -fno-rtti
)

create_cudaq_unittest_check_target("OptimizerUnitTests" "optimizer-unit-tests")

add_cudaq_unittest("AnalysisPassUnitTests" "analysis-passes"
    SOURCES
        QubitIdentityAnalysisTest.cpp
        UnitaryOpGroupingTest.cpp
    INCLUDES
        ${CMAKE_SOURCE_DIR}/cudaq/lib/Optimizer/Analysis
    LINKED_LIBS
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
        MLIRParser
        QuakeDialect
        CCDialect
        OptAnalysis
    COMPILE_OPTS
)

create_cudaq_unittest_check_target("AnalysisPassUnitTests" "analysis-passes")

add_cudaq_unittest("DeviceUnitTests" "device-unit-tests"
    SOURCES
        DeviceTest.cpp
    INCLUDES
    LINKED_LIBS
        MLIRIR
    COMPILE_OPTS
      -fno-rtti
)

create_cudaq_unittest_check_target("DeviceUnitTests" "device-unit-tests")

add_cudaq_unittest("test_quake_synth" "quake-synth"
    SOURCES
        QuakeSynthTester.cpp
    LINKED_LIBS
        cudaq
        cudaq-operator
        cudaq-mlir-runtime
        cudaq-builder
        cudaq-common
        cudaq-platform-default
        nvqir-qpp 
        nvqir
        cudaq-em-default
    LINK_OPTS
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<PLATFORM_ID:Darwin>>>:${CUDAQ_FORCE_LINK_FLAG}>
    COMPILE_OPTS
)
