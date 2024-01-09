# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(remote-simulator-qpu OBJECT RemoteSimulatorQPU.cpp)
target_link_libraries(remote-simulator-qpu PUBLIC 
                        cudaq-common 
                        cudaq-mlir-runtime
                        rest-remote-platform-client
                    )