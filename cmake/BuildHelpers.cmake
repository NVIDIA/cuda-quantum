# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard()

# If OpenMP is enabled and found, adds the necessary compile definitions to the
# given target, and the necessary dependencies to the given list of dependencies.
function(add_openmp_configurations TARGET_NAME DEPENDENCIES)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP Found. Adding build flags to target ${TARGET_NAME}: ${OpenMP_CXX_FLAGS}.")
        list(APPEND ${DEPENDENCIES} OpenMP::OpenMP_CXX)
        set(${DEPENDENCIES} "${${DEPENDENCIES}}" PARENT_SCOPE) 
        target_compile_definitions(${TARGET_NAME} PRIVATE HAS_OPENMP)
    elseif (CUDAQ_REQUIRE_OPENMP)
        message(FATAL_ERROR "OpenMP not found.")
    endif()
endfunction()

# If OpenMP is enabled and found, adds the necessary compile definitions to the
# interface dependencies of the given target.
function(add_openmp_interface_definitions TARGET_NAME)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP Found. Adding interface definitions to target ${TARGET_NAME}.")
        target_compile_definitions(${TARGET_NAME} INTERFACE HAS_OPENMP)
    elseif (CUDAQ_REQUIRE_OPENMP)
        message(FATAL_ERROR "OpenMP not found.")
    endif()
endfunction()

# macOS Two-Level Namespace Workaround: Force-load LLVM CodeGen libraries.
#
# Problem: macOS uses two-level namespace linking where each shared library
# has its own copy of static data. LLVM's TargetRegistry uses static initializers
# to register targets (X86, AArch64, etc.) into a global registry. Without
# force-loading, these registrations happen in the wrong library's copy of
# the registry, causing "target not found" errors during JIT compilation.
#
# Solution: -force_load ensures all symbols from these archives are included,
# triggering their static initializers in the correct library context.
function(add_lib_loading_macos_workaround TARGET_NAME NATIVE_TARGET_LIBS)
    if(APPLE)
        target_link_libraries(${TARGET_NAME} PRIVATE LLVMCodeGen)
        target_link_options(${TARGET_NAME} PRIVATE
            "-Wl,-force_load,$<TARGET_FILE:LLVMCodeGen>")

        if(NATIVE_TARGET_LIBS)
            target_link_libraries(${TARGET_NAME} PRIVATE
                LLVM${LLVM_NATIVE_ARCH}CodeGen
                LLVM${LLVM_NATIVE_ARCH}Info
                LLVM${LLVM_NATIVE_ARCH}Desc)
            target_link_options(${TARGET_NAME} PRIVATE
                "-Wl,-force_load,$<TARGET_FILE:LLVM${LLVM_NATIVE_ARCH}CodeGen>"
                "-Wl,-force_load,$<TARGET_FILE:LLVM${LLVM_NATIVE_ARCH}Info>"
                "-Wl,-force_load,$<TARGET_FILE:LLVM${LLVM_NATIVE_ARCH}Desc>")
        endif()
    endif()
endfunction()

# Making a NVQIR backend lib or config file available inside wheel
function(add_target_libs_to_wheel nvqir_backend_lib_or_config)
    if (NOT EXISTS "${nvqir_backend_lib_or_config}")
        message(FATAL_ERROR "Invalid file path to NVQIR backend lib or config: ${nvqir_backend_lib_or_config}.")
    endif()
    get_filename_component(FILE_EXTENSION ${nvqir_backend_lib_or_config} EXT)
    if ("${FILE_EXTENSION}" STREQUAL ".so")
        message(STATUS "Installing NVQIR backend lib '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION lib)
    elseif("${FILE_EXTENSION}" STREQUAL ".yml")
        message(STATUS "Installing NVQIR backend config '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION targets)
    else()
        message(WARNING "Unknown file extension of ${nvqir_backend_lib_or_config} file. It will be ignored.")
    endif()
endfunction()
