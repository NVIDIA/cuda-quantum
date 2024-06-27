# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-remote-simulator-qpu SHARED RemoteSimulatorQPU.cpp)
target_link_libraries(cudaq-remote-simulator-qpu PUBLIC 
                        cudaq
                        cudaq-common 
                        cudaq-mlir-runtime
                        rest-remote-platform-client
                        fmt::fmt-header-only
                        cudaq-platform-mqpu
                    )
                    
install(TARGETS cudaq-remote-simulator-qpu DESTINATION lib)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-remote-simulator-qpu-targets DESTINATION lib)
