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
# Note: this app linked against CUDA Quantum runtime (dynamic libraries).
# Hence, we don't enforce static linking. 
if (CMAKE_EXE_LINKER_FLAGS)
    # Convert it to a list
    string(REPLACE " " ";" LINKER_FLAGS_LIST ${CMAKE_EXE_LINKER_FLAGS}) 
    # Remove the "-static" flag
    list(REMOVE_ITEM LINKER_FLAGS_LIST -static)
    # Convert the list back to a space-separated string
    string(REPLACE ";" " " CMAKE_EXE_LINKER_FLAGS "${LINKER_FLAGS_LIST}") 
endif()

target_link_libraries(${TOOL_NAME} PRIVATE cudaq rest-remote-platform-server LLVMSupport)
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
install(TARGETS ${TOOL_NAME} DESTINATION bin)
