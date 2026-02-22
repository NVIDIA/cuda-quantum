# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-pasqal-qpu)
message(STATUS "Building Pasqal REST QPU.")

add_library(${LIBRARY_NAME}
  SHARED
    PasqalRemoteRESTQPU.cpp
    PasqalServerHelper.cpp
    PasqalQrmiServerHelper.cpp
    PasqalExecutor.cpp
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
    pthread
    cudaq-mlir-runtime 
    fmt::fmt-header-only
    cudaq 
    cudaq-platform-default
    nvqir
) 
    
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

add_target_config(pasqal)

if(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # TODO, right now build from source, but before merging this should become a
  # find_package for a released version of QRMI.
  set(QRMI_ROOT "${CMAKE_SOURCE_DIR}/../qrmi" CACHE PATH "Path to QRMI sources")
  find_path(QRMI_INCLUDE_DIR qrmi.h PATHS "${QRMI_ROOT}" NO_DEFAULT_PATH)
  find_library(QRMI_LIBRARY NAMES qrmi
               PATHS "${QRMI_ROOT}/target/release" "${QRMI_ROOT}/target/debug"
               NO_DEFAULT_PATH)

  if(QRMI_INCLUDE_DIR AND QRMI_LIBRARY)
    message(STATUS "Enabling Pasqal QRMI mode support in pasqal target.")
    target_include_directories(${LIBRARY_NAME} PRIVATE ${QRMI_INCLUDE_DIR})
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${QRMI_LIBRARY})
    target_compile_definitions(${LIBRARY_NAME}
                               PRIVATE CUDAQ_PASQAL_QRMI_ENABLED)

    get_filename_component(QRMI_LIBRARY_DIR "${QRMI_LIBRARY}" DIRECTORY)
    set_target_properties(${LIBRARY_NAME} PROPERTIES
      BUILD_RPATH "${QRMI_LIBRARY_DIR}"
      INSTALL_RPATH "${QRMI_LIBRARY_DIR}"
    )
  else()
    message(WARNING
      "Pasqal QRMI connector requested but qrmi.h/libqrmi not found. "
      "Building pasqal target without QRMI mode support.")
  endif()
endif()
