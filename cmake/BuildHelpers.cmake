# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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
