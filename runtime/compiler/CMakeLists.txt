# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME compiler)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(${LIBRARY_NAME} SHARED 
  compiler.cpp
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime> 
         $<INSTALL_INTERFACE:include> 
  PRIVATE .)

set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

# Private link to nvqir-qpp, it will always be here, 
# and by keeping it private, others can specify the
# backend library later on
target_link_libraries(${LIBRARY_NAME}
  PUBLIC cudaq-common )

install(TARGETS ${LIBRARY_NAME} 
        EXPORT compiler
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include/compiler)
