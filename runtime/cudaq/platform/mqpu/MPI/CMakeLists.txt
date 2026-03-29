# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(qpu-mpi-process-group OBJECT QpuProcessGroup.cpp)
target_link_libraries(qpu-mpi-process-group PUBLIC 
                        cudaq-common 
                        cudaq-logger 
)
target_include_directories(qpu-mpi-process-group PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)