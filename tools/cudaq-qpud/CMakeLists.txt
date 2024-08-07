# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
set(TOOL_NAME cudaq-qpud)
add_executable(${TOOL_NAME} RestServerMain.cpp)
set_target_properties(${TOOL_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
# Note: this app linked against CUDA-Q runtime (dynamic libraries).
# Hence, we don't enforce static linking. 
if (CMAKE_EXE_LINKER_FLAGS)
    # Convert it to a list
    string(REPLACE " " ";" LINKER_FLAGS_LIST ${CMAKE_EXE_LINKER_FLAGS}) 
    # Remove the "-static" flag
    list(REMOVE_ITEM LINKER_FLAGS_LIST -static)
    # Convert the list back to a space-separated string
    string(REPLACE ";" " " CMAKE_EXE_LINKER_FLAGS "${LINKER_FLAGS_LIST}") 
endif()
if (${CMAKE_EXE_LINKER_FLAGS} MATCHES "^.*static-libstdc\\+\\+.*")
    # If doing static linking to libstdc++, add the full archive to make sure all symbols are available 
    # (for later JIT compilation of incoming requests).
    target_link_libraries(${TOOL_NAME} PRIVATE
        -Wl,--whole-archive
        -lstdc++
        -Wl,--no-whole-archive
    )
endif()
target_link_libraries(${TOOL_NAME} PRIVATE cudaq rest-remote-platform-server cudaq-mlir-runtime)
target_compile_options(${TOOL_NAME} PRIVATE -fno-rtti)
export_executable_symbols_for_plugins(${TOOL_NAME})

install(TARGETS ${TOOL_NAME} DESTINATION bin)
install(FILES ${TOOL_NAME}.py DESTINATION bin)
install(FILES nvqc_proxy.py DESTINATION bin)
install(FILES json_request_runner.py DESTINATION bin)
