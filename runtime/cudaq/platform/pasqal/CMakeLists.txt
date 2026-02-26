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
  set(QRMI_ROOT "" CACHE PATH "Optional prefix for a prebuilt QRMI installation")

  # Example (disabled): fetch prebuilt QRMI artifacts from GitHub Releases.
  # Keep this commented out until release assets/policy are finalized.
  #
  # set(QRMI_VERSION "v0.12.0")
  # set(QRMI_ROOT "${CMAKE_BINARY_DIR}/qrmi-${QRMI_VERSION}")
  # execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${QRMI_ROOT}/include" "${QRMI_ROOT}/lib")
  # file(DOWNLOAD
  #   "https://github.com/qiskit-community/qrmi/releases/download/${QRMI_VERSION}/qrmi.h"
  #   "${QRMI_ROOT}/include/qrmi.h" SHOW_PROGRESS)
  # file(DOWNLOAD
  #   "https://github.com/qiskit-community/qrmi/releases/download/${QRMI_VERSION}/libqrmi.so"
  #   "${QRMI_ROOT}/lib/libqrmi.so" SHOW_PROGRESS)
  # Then configure with: -DQRMI_ROOT=${QRMI_ROOT}

  set(_qrmi_include_hints)
  set(_qrmi_library_hints)
  if(QRMI_ROOT)
    list(APPEND _qrmi_include_hints
      "${QRMI_ROOT}"
      "${QRMI_ROOT}/include")
    list(APPEND _qrmi_library_hints
      "${QRMI_ROOT}/lib"
      "${QRMI_ROOT}/lib64"
      "${QRMI_ROOT}/target/release"
      "${QRMI_ROOT}/target/debug")
  endif()

  find_path(QRMI_INCLUDE_DIR qrmi.h
            HINTS ${_qrmi_include_hints})
  find_library(QRMI_LIBRARY NAMES qrmi
               HINTS ${_qrmi_library_hints})

  if(NOT QRMI_INCLUDE_DIR OR NOT QRMI_LIBRARY)
    message(FATAL_ERROR
      "Pasqal QRMI connector requested but prebuilt qrmi.h/libqrmi was not found. "
      "Install QRMI first and optionally set -DQRMI_ROOT=<prefix>.")
  endif()

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${QRMI_INCLUDE_DIR}")
  set(_qrmi_version_check_source_lines
    "#include <qrmi.h>"
    "#if !defined(QRMI_VERSION) || (QRMI_VERSION < QRMI_VERSION_NUMERIC(0,12,0))"
    "#error QRMI version must be >= 0.12.0"
    "#endif"
    "int main() {}")
  string(JOIN "\n" _qrmi_version_check_source ${_qrmi_version_check_source_lines})
  check_cxx_source_compiles("${_qrmi_version_check_source}"
    CUDAQ_QRMI_VERSION_GTE_0_12_0)
  unset(CMAKE_REQUIRED_INCLUDES)

  if(NOT CUDAQ_QRMI_VERSION_GTE_0_12_0)
    message(FATAL_ERROR
      "Pasqal QRMI connector requires QRMI version >= 0.12.0.")
  endif()

  message(STATUS "Enabling Pasqal QRMI mode support in pasqal target.")
  target_include_directories(${LIBRARY_NAME} PRIVATE ${QRMI_INCLUDE_DIR})
  target_link_libraries(${LIBRARY_NAME} PRIVATE ${QRMI_LIBRARY})
  target_compile_definitions(${LIBRARY_NAME}
                             PRIVATE CUDAQ_PASQAL_QRMI_ENABLED)

  get_filename_component(QRMI_LIBRARY_DIR "${QRMI_LIBRARY}" DIRECTORY)
  set_target_properties(${LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "${QRMI_LIBRARY_DIR}"
  )
endif()
