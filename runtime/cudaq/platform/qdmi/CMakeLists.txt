# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-qdmi-qpu)

include(FetchContent)
set(CUDAQ_QDMI_MQT_CORE_REV
    "43047338aab5d85dbbd4c6bf88dcac31bb2eac74"
    CACHE STRING "MQT Core revision used by the QDMI backend")

block(SCOPE_FOR VARIABLES)
  set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
  set(ENABLE_IPO OFF)
  set(MQT_CORE_INSTALL OFF)
  set(BUILD_MQT_CORE_BINDINGS OFF)
  set(BUILD_MQT_CORE_TESTS OFF)
  set(BUILD_MQT_CORE_BENCHMARKS OFF)
  set(BUILD_MQT_CORE_DOCUMENTATION OFF)
  set(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE ${CUDAQ_BUILD_TESTS})
  set(BUILD_MQT_CORE_QDMI_SC_DEVICE ${CUDAQ_BUILD_TESTS})

  FetchContent_Declare(mqt_core
    GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
    GIT_TAG ${CUDAQ_QDMI_MQT_CORE_REV}
    GIT_SHALLOW FALSE
    SYSTEM
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(mqt_core)
endblock()

if (NOT TARGET MQT::CoreQDMI)
  message(FATAL_ERROR
    "The QDMI backend requires the MQT::CoreQDMI target from MQT Core.")
endif()

message(STATUS "Building QDMI QPU.")

add_library(${LIBRARY_NAME}
  SHARED
    QDMIQPU.cpp
)

target_include_directories(${LIBRARY_NAME} PRIVATE .
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-operator
    cudaq-common
  PRIVATE
    cudaq-logger
    cudaq-mlir-runtime
    fmt::fmt-header-only
    cudaq
    cudaq-platform-default
    CUDAQTargetConfigYaml
    MQT::CoreQDMI
    LLVMBitReader
    LLVMBitWriter
    LLVMCore
    LLVMSupport
    nvqir
)

if (UNIX AND NOT APPLE)
  target_link_options(${LIBRARY_NAME} PRIVATE "LINKER:--exclude-libs,ALL")
endif()

set(QDMI_QPU_LIBRARY
    "${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(qdmi.yml.in ${CMAKE_BINARY_DIR}/targets/qdmi.yml @ONLY)

if (CUDAQ_BUILD_TESTS)
  mqt_copy_qdmi_runtime(${LIBRARY_NAME}
    MQT::CoreQDMI_DDSIM_Device
    MQT::CoreQDMIScDevice)
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION lib COMPONENT Runtime)
install(FILES ${CMAKE_BINARY_DIR}/targets/qdmi.yml
        DESTINATION targets COMPONENT Runtime)
