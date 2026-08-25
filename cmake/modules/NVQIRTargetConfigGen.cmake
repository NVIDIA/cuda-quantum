# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

function(nvqir_write_target_config BackendName GPURequirements FilePath)
  file (WRITE ${FilePath} "name: ${BackendName}\ndescription: \"NVQIR ${BackendName} backend\"\ngpu-requirements: ${GPURequirements}\nconfig:\n\tnvqir-simulation-backend: ${BackendName}")
endfunction()
