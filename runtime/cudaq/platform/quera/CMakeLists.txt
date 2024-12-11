# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(cudaq-rest-qpu PRIVATE QuEraExecutor.cpp QuEraServerHelper.cpp)

add_library(cudaq-serverhelper-quera SHARED QuEraExecutor.cpp QuEraServerHelper.cpp )
target_link_libraries(cudaq-serverhelper-quera
  PUBLIC 
    cudaq-serverhelper-braket
    cudaq-common 
    fmt::fmt-header-only 
)
install(TARGETS cudaq-serverhelper-quera DESTINATION lib)

set(LIBRARY_NAME cudaq-quera-qpu)
message(STATUS "Building QuEra REST QPU.")

add_library(${LIBRARY_NAME} SHARED QuEraRemoteRESTQPU.cpp)

target_include_directories(${LIBRARY_NAME} PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-serverhelper-braket 
    cudaq-spin 
    cudaq-common 
  PRIVATE
    pthread
    cudaq-mlir-runtime 
    fmt::fmt-header-only
    cudaq 
    cudaq-platform-default)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(quera)
