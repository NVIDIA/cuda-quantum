# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-pasqal-qpu)
set(SERVERHELPER_LIBRARY_NAME cudaq-serverhelper-pasqal) # For some reason need to set this for runtime discoverability of the server helper symbols
message(STATUS "Building Pasqal REST QPU.")

add_library(${LIBRARY_NAME}
  SHARED
    PasqalRemoteRESTQPU.cpp
)

add_library(${SERVERHELPER_LIBRARY_NAME}
  SHARED
    PasqalServerHelper.cpp
    PasqalQrmiServerHelper.cpp
    PasqalExecutor.cpp
)

set(_pasqal_public_link_libs
  cudaq-operator
  cudaq-common)
set(_pasqal_private_link_libs
  pthread
  cudaq-mlir-runtime
  fmt::fmt-header-only
  cudaq
  cudaq-platform-default
  nvqir)
foreach(_pasqal_target IN ITEMS ${LIBRARY_NAME} ${SERVERHELPER_LIBRARY_NAME})
  target_include_directories(${_pasqal_target} PRIVATE .
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${_pasqal_target}
    PUBLIC ${_pasqal_public_link_libs}
    PRIVATE ${_pasqal_private_link_libs}
  )
endforeach()
    
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${SERVERHELPER_LIBRARY_NAME} DESTINATION lib)

add_target_config(pasqal)

if(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  macro(_pasqal_download_or_fail url out_path label)
    file(DOWNLOAD "${url}" "${out_path}" STATUS _pasqal_dl_status SHOW_PROGRESS)
    list(GET _pasqal_dl_status 0 _pasqal_dl_code)
    if(NOT _pasqal_dl_code EQUAL 0)
      list(GET _pasqal_dl_status 1 _pasqal_dl_error)
      message(FATAL_ERROR
        "Failed to download ${label} from ${url}: ${_pasqal_dl_error}")
    endif()
  endmacro()

  set(QRMI_ROOT "" CACHE PATH "Optional prefix for a prebuilt QRMI installation")
  set(CUDAQ_PASQAL_QRMI_FETCH_RELEASE OFF CACHE BOOL
      "Download QRMI prebuilt artifacts from GitHub release assets")
  set(CUDAQ_PASQAL_QRMI_RELEASE_TAG "v0.12.0" CACHE STRING
      "QRMI release tag used when CUDAQ_PASQAL_QRMI_FETCH_RELEASE=ON")

  if(CUDAQ_PASQAL_QRMI_FETCH_RELEASE AND NOT QRMI_ROOT)
    set(QRMI_ROOT "${CMAKE_BINARY_DIR}/qrmi-${CUDAQ_PASQAL_QRMI_RELEASE_TAG}"
        CACHE PATH "Auto-downloaded QRMI prefix" FORCE)
    set(_qrmi_release_base
        "https://github.com/qiskit-community/qrmi/releases/download/${CUDAQ_PASQAL_QRMI_RELEASE_TAG}")

    file(MAKE_DIRECTORY "${QRMI_ROOT}/include" "${QRMI_ROOT}/lib")
    _pasqal_download_or_fail("${_qrmi_release_base}/qrmi.h"
      "${QRMI_ROOT}/include/qrmi.h" "qrmi.h")
    _pasqal_download_or_fail("${_qrmi_release_base}/libqrmi.so"
      "${QRMI_ROOT}/lib/libqrmi.so" "libqrmi.so")
    _pasqal_download_or_fail("${_qrmi_release_base}/libqrmi.sha256"
      "${QRMI_ROOT}/lib/libqrmi.sha256" "libqrmi.sha256")

    file(READ "${QRMI_ROOT}/lib/libqrmi.sha256" _qrmi_sha_line)
    string(REGEX MATCH "^[0-9A-Fa-f]+" _qrmi_expected_sha "${_qrmi_sha_line}")
    if(NOT _qrmi_expected_sha)
      message(FATAL_ERROR
        "Failed to parse expected SHA256 from ${QRMI_ROOT}/lib/libqrmi.sha256")
    endif()
    file(SHA256 "${QRMI_ROOT}/lib/libqrmi.so" _qrmi_actual_sha)
    string(TOLOWER "${_qrmi_expected_sha}" _qrmi_expected_sha)
    string(TOLOWER "${_qrmi_actual_sha}" _qrmi_actual_sha)
    if(NOT _qrmi_expected_sha STREQUAL _qrmi_actual_sha)
      message(FATAL_ERROR
        "QRMI checksum mismatch for downloaded libqrmi.so")
    endif()

    message(STATUS
      "Downloaded QRMI release assets into ${QRMI_ROOT}")
  endif()

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
  target_include_directories(${SERVERHELPER_LIBRARY_NAME} PRIVATE ${QRMI_INCLUDE_DIR})
  target_link_libraries(${SERVERHELPER_LIBRARY_NAME} PRIVATE ${QRMI_LIBRARY})
  target_compile_definitions(${SERVERHELPER_LIBRARY_NAME}
                             PRIVATE CUDAQ_PASQAL_QRMI_ENABLED)

  get_filename_component(QRMI_LIBRARY_DIR "${QRMI_LIBRARY}" DIRECTORY)
  set_target_properties(${SERVERHELPER_LIBRARY_NAME} PROPERTIES
    BUILD_RPATH "${QRMI_LIBRARY_DIR}"
  )
endif()
