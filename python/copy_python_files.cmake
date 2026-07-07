# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Copies the python source tree into the build directory for the
# CopyPythonFiles target. Excludes cudaq/mlir: the MLIR python machinery
# stages that subtree via symlink rules, and a second producer races them
# ("failed to create symbolic link ...: File exists").

if(NOT SOURCE_DIR OR NOT DESTINATION_DIR)
    message(FATAL_ERROR "SOURCE_DIR and DESTINATION_DIR must be defined")
endif()

file(COPY "${SOURCE_DIR}" DESTINATION "${DESTINATION_DIR}"
     REGEX "/cudaq/mlir(/|$)" EXCLUDE)
