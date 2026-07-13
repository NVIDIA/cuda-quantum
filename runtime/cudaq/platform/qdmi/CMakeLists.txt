# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-qdmi)

set(CUDAQ_QDMI_DDSIM_AVAILABLE OFF CACHE INTERNAL
  "Whether the QDMI DDSIM device is available for target tests" FORCE)
set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY "" CACHE INTERNAL
  "QDMI DDSIM device library for target tests" FORCE)

find_package(mqt-core CONFIG QUIET)
if (NOT TARGET MQT::CoreFoMaC)
  return()
endif()

message(STATUS "Building QDMI platform.")

add_library(${LIBRARY_NAME}
  SHARED
    QDMIPlatform.cpp
    QDMIQPU.cpp
)

if (TARGET MQT::CoreQDMI_DDSIM_Device)
  get_target_property(_qdmi_ddsim_device_library
    MQT::CoreQDMI_DDSIM_Device IMPORTED_LOCATION_RELEASE)
  if (NOT _qdmi_ddsim_device_library)
    get_target_property(_qdmi_ddsim_device_library
      MQT::CoreQDMI_DDSIM_Device IMPORTED_LOCATION)
  endif()
  if (EXISTS "${_qdmi_ddsim_device_library}")
    set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY
      "${_qdmi_ddsim_device_library}" CACHE INTERNAL
      "QDMI DDSIM device library for target tests" FORCE)
    set(CUDAQ_QDMI_DDSIM_AVAILABLE ON CACHE INTERNAL
      "Whether the QDMI DDSIM device is available for target tests" FORCE)
  endif()
endif()

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
    CUDAQTargetConfigYaml
    MQT::CoreFoMaC
    nvqir
)

add_target_config(qdmi)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
