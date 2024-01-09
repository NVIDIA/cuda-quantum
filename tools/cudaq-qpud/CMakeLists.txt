# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(TOOL_NAME cudaq-qpud)
add_executable(${TOOL_NAME} RestServerMain.cpp)
set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
target_link_libraries(${TOOL_NAME} PRIVATE cudaq rest-remote-platform-server LLVMSupport)
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
install(TARGETS ${TOOL_NAME} DESTINATION bin)
