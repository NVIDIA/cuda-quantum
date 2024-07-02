# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
add_subdirectory(default)
if (OPENSSL_FOUND)
  add_subdirectory(orca)
endif()
if (CUDAQ_ENABLE_REST)
  add_subdirectory(mqpu)
endif()
