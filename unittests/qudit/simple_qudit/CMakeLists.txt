# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-em-qudit)

add_library(${LIBRARY_NAME} SHARED SimpleQuditExecutionManager.cpp)

target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
      $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC cudaq-operator PRIVATE cudaq-common fmt::fmt-header-only libqpp)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
