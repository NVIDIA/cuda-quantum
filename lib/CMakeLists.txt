# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  add_subdirectory(Frontend)
endif()
add_subdirectory(Optimizer)
add_subdirectory(Support)
