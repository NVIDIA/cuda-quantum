# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE OQCServerHelp.cpp)

add_library(cudaq-serverhelper-oqc SHARED OQCServerHelp.cpp )
target_link_libraries(cudaq-serverhelper-oqc
  PUBLIC 
    cudaq-logger 
    cudaq-common 
)
install(TARGETS cudaq-serverhelper-oqc DESTINATION lib
        COMPONENT Runtime)

add_target_mapping_arch(oqc lucy.txt)
add_target_mapping_arch(oqc toshiko.txt)
