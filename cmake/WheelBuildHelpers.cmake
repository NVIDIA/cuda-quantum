# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard()

# Making a NVQIR backend lib or config file available inside wheel 
function(add_target_libs_to_wheel nvqir_backend_lib_or_config)
    if (NOT EXISTS "${nvqir_backend_lib_or_config}")
        message(FATAL_ERROR "Invalid file path to NVQIR backend lib or config: ${nvqir_backend_lib_or_config}.")
    endif()
    get_filename_component(FILE_EXTENSION ${nvqir_backend_lib_or_config} EXT)
    if ("${FILE_EXTENSION}" STREQUAL ".so")
        message(STATUS "Installing NVQIR backend lib '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION lib)
    elseif("${FILE_EXTENSION}" STREQUAL ".config")
        message(STATUS "Installing NVQIR backend config '${nvqir_backend_lib_or_config}'")
        install(FILES ${nvqir_backend_lib_or_config} DESTINATION targets)
    else()
        message(WARNING "Unknown file extension of ${nvqir_backend_lib_or_config} file. It will be ignored.")
    endif()
endfunction()
