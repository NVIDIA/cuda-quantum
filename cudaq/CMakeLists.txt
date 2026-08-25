# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (NOT DEFINED CUDAQ_ENABLE_CORE)
  set(CUDAQ_ENABLE_CORE ON CACHE BOOL "Enable building CUDA-Q core.")
endif()

if(CUDAQ_ENABLE_CORE)
  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(tools)
  if(CUDAQ_BUILD_TESTS AND NOT CUDAQ_DISABLE_CPP_FRONTEND)
    add_subdirectory(test)
  endif()
endif()
