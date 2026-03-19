# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(cudaq-pass-pipeline-logging
  PassPipelineLogging.cpp

  LINK_LIBS PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
)

target_include_directories(cudaq-pass-pipeline-logging
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime
)

target_compile_options(cudaq-pass-pipeline-logging PRIVATE -fno-rtti)

install(TARGETS cudaq-pass-pipeline-logging DESTINATION lib)
