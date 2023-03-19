# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# If we have CUDA, then we want to be able to 
# call cudaSetDevice.
if(CUDA_FOUND)
  target_sources(qpud PRIVATE NvidiaPlatformHelper.cu)
  target_link_libraries(qpud PRIVATE ${CUDA_LIBRARIES})
else()
  message(STATUS "CUDA Not Found, QPUD using FakeNvidiaPlatformHelper")
  target_sources(qpud PRIVATE FakeNvidiaPlatformHelper.cpp)
endif()