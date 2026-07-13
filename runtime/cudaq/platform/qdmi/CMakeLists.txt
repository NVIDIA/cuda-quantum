# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-qdmi)

# Lit substitutions expect canonical CMake boolean values (0 or 1).
set(CUDAQ_QDMI_DDSIM_AVAILABLE 0 CACHE INTERNAL
  "Whether the QDMI DDSIM device is available for target tests" FORCE)
set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY "" CACHE INTERNAL
  "QDMI DDSIM device library for target tests" FORCE)

find_package(mqt-core CONFIG REQUIRED)
if (NOT TARGET MQT::CoreFoMaC)
  message(FATAL_ERROR
    "The QDMI backend requires the MQT::CoreFoMaC target from MQT Core.")
endif()

message(STATUS "Building QDMI platform.")

add_library(${LIBRARY_NAME}
  SHARED
    QDMIPlatform.cpp
    QDMIQPU.cpp
)

if (TARGET MQT::CoreQDMI_DDSIM_Device)
  get_target_property(_qdmi_ddsim_configs
    MQT::CoreQDMI_DDSIM_Device IMPORTED_CONFIGURATIONS)
  foreach(_qdmi_ddsim_config IN LISTS _qdmi_ddsim_configs)
    string(TOUPPER "${_qdmi_ddsim_config}" _qdmi_ddsim_config_upper)
    get_target_property(_qdmi_ddsim_device_library
      MQT::CoreQDMI_DDSIM_Device
      "IMPORTED_LOCATION_${_qdmi_ddsim_config_upper}")
    if (_qdmi_ddsim_device_library)
      break()
    endif()
  endforeach()
  if (NOT _qdmi_ddsim_device_library)
    get_target_property(_qdmi_ddsim_device_library
      MQT::CoreQDMI_DDSIM_Device IMPORTED_LOCATION)
  endif()
  if (EXISTS "${_qdmi_ddsim_device_library}")
    set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY
      "${_qdmi_ddsim_device_library}" CACHE INTERNAL
      "QDMI DDSIM device library for target tests" FORCE)
    set(CUDAQ_QDMI_DDSIM_AVAILABLE 1 CACHE INTERNAL
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
