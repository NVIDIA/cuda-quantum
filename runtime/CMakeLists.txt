# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(logger)
add_subdirectory(internal)
add_subdirectory(common)
add_subdirectory(nvqir)
add_subdirectory(cudaq)
add_subdirectory(test)

# Install
# ==============================================================================

install(DIRECTORY cudaq
        DESTINATION include
        COMPONENT Development
        FILES_MATCHING PATTERN "*.h"
        PATTERN "device_call_service.h" EXCLUDE
        PATTERN "nlopt-src" EXCLUDE)
install(DIRECTORY include/cudaq
        DESTINATION include
        COMPONENT Development
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY common
        DESTINATION include
        COMPONENT Development
        FILES_MATCHING PATTERN "*.h")
install(FILES nvqir/AnalysisScope.h
               nvqir/CircuitSimulator.h
               nvqir/Gates.h
               nvqir/QIRTypes.h
        DESTINATION include/nvqir
        COMPONENT Development)
install(FILES nvqir/resourcecounter/ResourceCounterScope.h
        DESTINATION include/nvqir/resourcecounter
        COMPONENT Development)
install(FILES nvqir/dem/DemScope.h
        DESTINATION include/nvqir/dem
        COMPONENT Development)
install(FILES cudaq.h DESTINATION include
        COMPONENT Development)
