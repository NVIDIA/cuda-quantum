# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(common)
add_subdirectory(nvqir)
add_subdirectory(cudaq)

# Install
# ==============================================================================

install (DIRECTORY cudaq DESTINATION include 
            FILES_MATCHING PATTERN "*.h" 
            PATTERN "nlopt-src" EXCLUDE)
install (DIRECTORY common DESTINATION include FILES_MATCHING PATTERN "*.h")
install (FILES nvqir/CircuitSimulator.h
               nvqir/QIRTypes.h
               nvqir/Gates.h
        DESTINATION include/nvqir)
install (FILES cudaq.h DESTINATION include)
