# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-default-qpud)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_DEFAULTPLATFORM_SRC
  DefaultQPUDQuantumPlatform.cpp
  ../common/QuantumExecutionQueue.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_DEFAULTPLATFORM_SRC})
target_include_directories(${LIBRARY_NAME}
  PUBLIC . ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/runtime )
target_link_libraries(${LIBRARY_NAME}
  PUBLIC fmt::fmt-header-only cudaq-qpud-client PRIVATE fmt::fmt-header-only rpc LLVMSupport)

cudaq_library_set_rpath(${LIBRARY_NAME})

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
