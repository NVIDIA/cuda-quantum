# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(CUDAQ_ENABLE_TII_BACKEND)
  target_sources(cudaq-rest-qpu PRIVATE TiiServerHelper.cpp)
  add_target_config(tii)

  add_library(cudaq-serverhelper-tii SHARED TiiServerHelper.cpp)
  target_link_libraries(cudaq-serverhelper-tii
    PUBLIC
    cudaq-common
    cudaq-logger
  )
  install(TARGETS cudaq-serverhelper-tii DESTINATION lib)
endif()
