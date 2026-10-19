# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
  CUDAQ_EDEM_STANDALONE_BUILD)
set(CUDAQ_EDEM_VERSION 0.1.0)
set(CUDAQ_EDEM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(CUDAQ_EDEM_STANDALONE_BUILD)
  project(cudaq-edem VERSION ${CUDAQ_EDEM_VERSION} LANGUAGES C)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  # Match CUDA-Q's default library directory while honoring explicit overrides.
  set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Object code libraries")
elseif(NOT DEFINED BUILD_SHARED_LIBS)
  # This default is local to this directory and its children.
  set(BUILD_SHARED_LIBS ON)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CUDAQ_EDEM_BUILD_TESTS "Build EDEM packaging tests"
  ${CUDAQ_EDEM_STANDALONE_BUILD})

if(NOT TARGET cudaq-gf2)
  add_subdirectory(../math/gf2 "${CMAKE_CURRENT_BINARY_DIR}/gf2")
endif()

add_library(cudaq-edem lib/edem.c)
add_library(cudaq::edem ALIAS cudaq-edem)
target_sources(cudaq-edem PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES include/cudaq/edem/core.h
)
target_link_libraries(cudaq-edem PUBLIC cudaq::gf2)
target_compile_features(cudaq-edem PUBLIC c_std_11)
set_target_properties(cudaq-edem PROPERTIES
  EXPORT_NAME edem
  C_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

# Both targets belong to one versioned distribution.
set_target_properties(cudaq-edem cudaq-gf2 PROPERTIES
  VERSION "${CUDAQ_EDEM_VERSION}"
  SOVERSION 0
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH_USE_LINK_PATH FALSE
)
if(APPLE)
  set_target_properties(cudaq-edem cudaq-gf2 PROPERTIES
    INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(cudaq-edem cudaq-gf2 PROPERTIES
    INSTALL_RPATH "$ORIGIN")
endif()

set(CUDAQ_EDEM_LIBRARY_COMPONENT Runtime)
set(CUDAQ_EDEM_DEVELOPMENT_COMPONENT Development)
if(DEFINED CUDAQ_CORE_LIBRARY_COMPONENT)
  set(CUDAQ_EDEM_LIBRARY_COMPONENT "${CUDAQ_CORE_LIBRARY_COMPONENT}")
endif()
if(DEFINED CUDAQ_CORE_DEVELOPMENT_COMPONENT)
  set(CUDAQ_EDEM_DEVELOPMENT_COMPONENT "${CUDAQ_CORE_DEVELOPMENT_COMPONENT}")
endif()

# GF2 defines its target; EDEM owns installation and packaging of both libraries.
install(TARGETS cudaq-edem cudaq-gf2
  EXPORT cudaq-edem-targets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT "${CUDAQ_EDEM_LIBRARY_COMPONENT}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${CUDAQ_EDEM_LIBRARY_COMPONENT}"
    NAMELINK_COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}"
  FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}"
)

set(CUDAQ_EDEM_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-edem")
configure_package_config_file(
  cmake/cudaq-edem-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-edem-config.cmake"
  INSTALL_DESTINATION "${CUDAQ_EDEM_CMAKE_INSTALL_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-edem-config-version.cmake"
  VERSION "${CUDAQ_EDEM_VERSION}"
  COMPATIBILITY SameMinorVersion)

install(EXPORT cudaq-edem-targets
  FILE cudaq-edem-targets.cmake
  NAMESPACE cudaq::
  DESTINATION "${CUDAQ_EDEM_CMAKE_INSTALL_DIR}"
  COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-edem-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-edem-config-version.cmake"
  DESTINATION "${CUDAQ_EDEM_CMAKE_INSTALL_DIR}"
  COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}")
install(FILES ../LICENSE ../NOTICE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/cudaq-edem"
  COMPONENT "${CUDAQ_EDEM_DEVELOPMENT_COMPONENT}")

if(CUDAQ_EDEM_BUILD_TESTS)
  enable_language(CXX)
  enable_testing()
  add_subdirectory(test)
endif()
