# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-logger
  SHARED
    Logger.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-logger)

target_include_directories(cudaq-logger
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<INSTALL_INTERFACE:include>
  PRIVATE .
    ${CMAKE_SOURCE_DIR}/tpls/spdlog/include
    ${CMAKE_SOURCE_DIR}/tpls/fmt/include)

target_link_libraries(cudaq-logger
  PRIVATE
    spdlog::spdlog
)

install(TARGETS cudaq-logger EXPORT cudaq-common-targets DESTINATION lib)
