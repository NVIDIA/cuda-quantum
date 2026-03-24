# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(mpi-decorated-qpu OBJECT MpiDecoratedQPU.cpp)
target_link_libraries(mpi-decorated-qpu PUBLIC 
                        cudaq-common 
                        cudaq-logger 
)
target_include_directories(mpi-decorated-qpu PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)