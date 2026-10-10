# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_unittest("ASTResultVisitorUnitTests" "ast-result-visitor"
    SOURCES
        ASTResultVisitorTester.cpp
    INCLUDES
        ${CMAKE_SOURCE_DIR}/cudaq/include
    LINKED_LIBS
        clangTooling
        clangFrontend
        clangAST
        clangBasic
        clangLex
        clangSerialization
    COMPILE_OPTS
      -fno-rtti
)

create_cudaq_unittest_check_target("ASTResultVisitorUnitTests" "ast-result-visitor")
