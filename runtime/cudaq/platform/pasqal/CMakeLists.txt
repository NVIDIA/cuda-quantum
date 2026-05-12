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

set(_serverhelper_sources
  PasqalServerHelper.cpp
  PasqalExecutor.cpp
)
if(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND _serverhelper_sources PasqalQrmiServerHelper.cpp)
endif()

add_library(${SERVERHELPER_LIBRARY_NAME}
  SHARED
    ${_serverhelper_sources}
)

set(_pasqal_public_link_libs
  cudaq-operator
  cudaq-common)
foreach(_pasqal_target IN ITEMS ${LIBRARY_NAME} ${SERVERHELPER_LIBRARY_NAME})
  target_include_directories(${_pasqal_target} PRIVATE .
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${_pasqal_target}
    PUBLIC ${_pasqal_public_link_libs}
  )
endforeach()

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    cudaq-mlir-runtime
    fmt::fmt-header-only
    cudaq
    cudaq-platform-default
    nvqir)

target_link_libraries(${SERVERHELPER_LIBRARY_NAME}
  PRIVATE
    fmt::fmt-header-only
    cudaq-logger
    pthread)
    
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${SERVERHELPER_LIBRARY_NAME} DESTINATION lib)

add_target_config(pasqal)

if(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_qrmi_include_hints)
  set(_qrmi_library_hints)
  if(QRMI_INSTALL_PREFIX)
    list(APPEND _qrmi_include_hints
      "${QRMI_INSTALL_PREFIX}"
      "${QRMI_INSTALL_PREFIX}/include")
    list(APPEND _qrmi_library_hints
      "${QRMI_INSTALL_PREFIX}/lib"
      "${QRMI_INSTALL_PREFIX}/lib64"
      "${QRMI_INSTALL_PREFIX}/target/release"
      "${QRMI_INSTALL_PREFIX}/target/debug")
  endif()

  find_path(QRMI_INCLUDE_DIR qrmi.h
            HINTS ${_qrmi_include_hints})
  find_library(QRMI_LIBRARY NAMES qrmi
               HINTS ${_qrmi_library_hints})

  if(NOT QRMI_INCLUDE_DIR OR NOT QRMI_LIBRARY)
    message(WARNING
      "Pasqal QRMI connector requested but qrmi.h/libqrmi was not found. "
      "Install QRMI and set QRMI_INSTALL_PREFIX=<prefix> if needed; "
      "building pasqal target without QRMI mode support.")
  else()
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
      message(WARNING
        "Pasqal QRMI connector requires QRMI version >= 0.12.0; "
        "building pasqal target without QRMI mode support.")
    else()
      message(STATUS "Enabling Pasqal QRMI mode support in pasqal target.")
      target_include_directories(${SERVERHELPER_LIBRARY_NAME} PRIVATE ${QRMI_INCLUDE_DIR})
      target_link_libraries(${SERVERHELPER_LIBRARY_NAME} PRIVATE ${QRMI_LIBRARY})
      target_compile_definitions(${SERVERHELPER_LIBRARY_NAME}
                                PRIVATE CUDAQ_PASQAL_QRMI_ENABLED)

      get_filename_component(QRMI_LIBRARY_DIR "${QRMI_LIBRARY}" DIRECTORY)
      set_property(TARGET ${SERVERHELPER_LIBRARY_NAME} APPEND PROPERTY
        BUILD_RPATH "${QRMI_LIBRARY_DIR}")
      set_property(TARGET ${SERVERHELPER_LIBRARY_NAME} APPEND PROPERTY
        INSTALL_RPATH "$ORIGIN" "${QRMI_LIBRARY_DIR}")
    endif()
  endif()
endif()
