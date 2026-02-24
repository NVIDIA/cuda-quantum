# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(SCALEWAY_SOURCES
  qio/Compression.cpp
  qio/Base64.cpp
  qio/QuantumProgram.cpp
  qio/QuantumProgramResult.cpp
  qio/QuantumComputationParameters.cpp
  qio/QuantumComputationModel.cpp
  qaas/V1Alpha1Client.cpp
  ScalewayServerHelper.cpp
)

target_sources(cudaq-rest-qpu
              PRIVATE
              ${SCALEWAY_SOURCES}
)

add_target_config(scaleway)

add_library(cudaq-serverhelper-scaleway
            SHARED
            ${SCALEWAY_SOURCES}
)

target_link_libraries(cudaq-serverhelper-scaleway
  PUBLIC
    cudaq-common
    cudaq-logger
    llvm-16-dev
)

install(TARGETS cudaq-serverhelper-scaleway DESTINATION lib)

