# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-qdmi-qpu)
set(SERVERHELPER_LIBRARY_NAME cudaq-serverhelper-qdmi)

set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY "" CACHE INTERNAL
  "QDMI DDSIM device library for target tests" FORCE)
set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY_DIR "" CACHE INTERNAL
  "QDMI DDSIM device library directory for target tests" FORCE)

find_package(mqt-core CONFIG QUIET)
if (NOT TARGET MQT::CoreFoMaC)
  return()
endif()

message(STATUS "Building QDMI QPU.")

add_library(${LIBRARY_NAME}
  SHARED
    QDMIQPU.cpp
    QDMIExecutor.cpp
    QDMIServerHelper.cpp
)

add_library(${SERVERHELPER_LIBRARY_NAME}
  SHARED
    QDMIExecutor.cpp
    QDMIServerHelper.cpp
)

if (TARGET MQT::CoreQDMI_DDSIM_Device)
  get_target_property(_qdmi_ddsim_device_library
    MQT::CoreQDMI_DDSIM_Device IMPORTED_LOCATION_RELEASE)
  if (NOT _qdmi_ddsim_device_library)
    get_target_property(_qdmi_ddsim_device_library
      MQT::CoreQDMI_DDSIM_Device IMPORTED_LOCATION)
  endif()
  if (EXISTS "${_qdmi_ddsim_device_library}")
    get_filename_component(_qdmi_ddsim_device_library_dir
      "${_qdmi_ddsim_device_library}" DIRECTORY)
    set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY
      "${_qdmi_ddsim_device_library}" CACHE INTERNAL
      "QDMI DDSIM device library for target tests" FORCE)
    set(CUDAQ_QDMI_DDSIM_DEVICE_LIBRARY_DIR
      "${_qdmi_ddsim_device_library_dir}" CACHE INTERNAL
      "QDMI DDSIM device library directory for target tests" FORCE)
  endif()
endif()

foreach(_qdmi_target IN ITEMS ${LIBRARY_NAME} ${SERVERHELPER_LIBRARY_NAME})
  target_include_directories(${_qdmi_target} PRIVATE .
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${_qdmi_target}
    PRIVATE
      cudaq-common
      cudaq-logger
      MQT::CoreFoMaC
  )
endforeach()

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-operator
    cudaq-common
  PRIVATE
    cudaq-mlir-runtime
    fmt::fmt-header-only
    cudaq
    cudaq-platform-default
    nvqir
)

add_target_config(qdmi)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${SERVERHELPER_LIBRARY_NAME} DESTINATION lib)
