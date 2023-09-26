# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard()

# Making a NVQIR backend lib available inside wheel 
function(add_target_libs_to_wheel nvqir_backend_lib nvqir_backend_config)
    message(STATUS "Installing NVQIR backend lib '${nvqir_backend_lib}' with config '${nvqir_backend_config}'")
    if (NOT EXISTS "${nvqir_backend_lib}" OR NOT EXISTS "${nvqir_backend_config}")
        message(FATAL_ERROR "Invalid file paths to NVQIR backend lib.")
    endif()
    install(FILES ${nvqir_backend_lib} DESTINATION lib)
    install(FILES ${nvqir_backend_config} DESTINATION targets)
endfunction()
