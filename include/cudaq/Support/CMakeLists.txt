# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Files needed at runtime
install (FILES TargetConfig.h Version.h DESTINATION include/cudaq/Support)

# Make files available to nvq++ tests running in the build directory
# cudaq-quake will use this as a fallback location
file(COPY TargetConfig.h Version.h DESTINATION ${CMAKE_BINARY_DIR}/include/cudaq/Support)
