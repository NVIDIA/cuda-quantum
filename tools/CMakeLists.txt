# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(cudaq-opt)
add_subdirectory(cudaq-translate)

if (NOT CUDAQ_DISABLE_CPP_FRONTEND)
  add_subdirectory(cudaq-lsp-server)
  add_subdirectory(cudaq-quake)
  add_subdirectory(fixup-linkage)
  add_subdirectory(nvqpp)
endif()
