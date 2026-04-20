# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
add_library(cudaq-remote-simulator-qpu SHARED RemoteSimulatorQPU.cpp)
target_link_libraries(cudaq-remote-simulator-qpu
  PUBLIC
    cudaq
    cudaq-common
    cudaq-logger
    cudaq-mlir-runtime
    rest-remote-platform-client
    cudaq-platform-mqpu
)
# rest-remote-platform-client only registers its symbols via static
# constructors; --as-needed drops it because no symbol is referenced
# directly. Bracket just that library with --no-as-needed so it stays
# in DT_NEEDED; otherwise the "rest" RemoteRuntimeClient is never
# registered and BaseRemoteSimulatorQPU's constructor segfaults on
# registry::get<RemoteRuntimeClient>("rest").
target_link_options(cudaq-remote-simulator-qpu PRIVATE
  "LINKER:--push-state,--no-as-needed"
  "LINKER:$<TARGET_FILE:rest-remote-platform-client>"
  "LINKER:--pop-state")
                    
install(TARGETS cudaq-remote-simulator-qpu DESTINATION lib)
endif()
