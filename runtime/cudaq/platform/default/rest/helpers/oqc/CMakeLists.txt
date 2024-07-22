# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/oqc.yml
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/oqc.yml)

target_sources(cudaq-rest-qpu PRIVATE OQCServerHelp.cpp)
add_target_config(oqc)

add_library(cudaq-serverhelper-oqc SHARED OQCServerHelp.cpp )
target_link_libraries(cudaq-serverhelper-oqc
  PUBLIC 
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-oqc DESTINATION lib)

add_target_mapping_arch(oqc lucy.txt)
add_target_mapping_arch(oqc toshiko.txt)
