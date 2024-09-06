# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-variable -Wno-error=unused-but-set-variable")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(STIM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/tpls/Stim)
set(STIM_BINARY_DIR ${CMAKE_BINARY_DIR}/tpls/Stim)

add_subdirectory(${STIM_SOURCE_DIR} ${STIM_BINARY_DIR})

macro (AddStimBackend LIBRARY_NAME SOURCE_FILE)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILE})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

  set (STIM_DEPENDENCIES "")
  list(APPEND STIM_DEPENDENCIES fmt::fmt-header-only cudaq-common)

  target_include_directories(${LIBRARY_NAME}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
        $<INSTALL_INTERFACE:include>)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC libstim
    PRIVATE ${STIM_DEPENDENCIES})

  set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()

AddStimBackend(nvqir-stim StimCircuitSimulator.cpp)

add_target_config(stim)
