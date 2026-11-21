# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.23)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
  CUDAQ_GADGETS_STANDALONE_BUILD)
set(CUDAQ_GADGETS_VERSION 0.1.0)

if(CUDAQ_GADGETS_STANDALONE_BUILD)
  project(cudaq-gadgets VERSION ${CUDAQ_GADGETS_VERSION} LANGUAGES C CXX)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Object code libraries")
  endif()
elseif(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_language(CXX)

option(CUDAQ_GADGETS_BUILD_TESTS "Build cudaq-gadgets component tests"
  ${CUDAQ_GADGETS_STANDALONE_BUILD})

if(NOT TARGET cudaq-edem)
  add_subdirectory(../edem "${CMAKE_CURRENT_BINARY_DIR}/edem")
endif()

add_library(cudaq-gadgets lib/core.cpp lib/analysis/analysis.cpp
  lib/circuit/circuit.cpp lib/circuit/pauli.cpp lib/circuit/standard.cpp
  lib/noise/noise.cpp)
add_library(cudaq::gadgets ALIAS cudaq-gadgets)
target_sources(cudaq-gadgets PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES include/cudaq/gadgets/core.h
    include/cudaq/gadgets/analysis/analysis.h
    include/cudaq/gadgets/circuit/circuit.h
    include/cudaq/gadgets/circuit/pauli.h
    include/cudaq/gadgets/circuit/standard.h
    include/cudaq/gadgets/gadget/gadget.h
    include/cudaq/gadgets/noise/noise.h
)
target_link_libraries(cudaq-gadgets PUBLIC cudaq::edem)
target_compile_features(cudaq-gadgets PUBLIC c_std_11)
target_compile_features(cudaq-gadgets PRIVATE cxx_std_20)
set_target_properties(cudaq-gadgets PROPERTIES
  EXPORT_NAME gadgets
  C_EXTENSIONS OFF
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  VERSION "${CUDAQ_GADGETS_VERSION}"
  SOVERSION 0
)

set(CUDAQ_GADGETS_LIBRARY_COMPONENT Runtime)
set(CUDAQ_GADGETS_DEVELOPMENT_COMPONENT Development)
if(DEFINED CUDAQ_CORE_LIBRARY_COMPONENT)
  set(CUDAQ_GADGETS_LIBRARY_COMPONENT "${CUDAQ_CORE_LIBRARY_COMPONENT}")
endif()
if(DEFINED CUDAQ_CORE_DEVELOPMENT_COMPONENT)
  set(CUDAQ_GADGETS_DEVELOPMENT_COMPONENT "${CUDAQ_CORE_DEVELOPMENT_COMPONENT}")
endif()

install(TARGETS cudaq-gadgets
  EXPORT cudaq-gadgets-targets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT "${CUDAQ_GADGETS_LIBRARY_COMPONENT}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${CUDAQ_GADGETS_LIBRARY_COMPONENT}"
    NAMELINK_COMPONENT "${CUDAQ_GADGETS_DEVELOPMENT_COMPONENT}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT "${CUDAQ_GADGETS_DEVELOPMENT_COMPONENT}"
  FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "${CUDAQ_GADGETS_DEVELOPMENT_COMPONENT}"
)

set(CUDAQ_GADGETS_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-gadgets")
configure_package_config_file(
  cmake/cudaq-gadgets-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-gadgets-config.cmake"
  INSTALL_DESTINATION "${CUDAQ_GADGETS_CMAKE_INSTALL_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-gadgets-config-version.cmake"
  VERSION "${CUDAQ_GADGETS_VERSION}"
  COMPATIBILITY SameMinorVersion)
install(EXPORT cudaq-gadgets-targets
  FILE cudaq-gadgets-targets.cmake
  NAMESPACE cudaq::
  DESTINATION "${CUDAQ_GADGETS_CMAKE_INSTALL_DIR}"
  COMPONENT "${CUDAQ_GADGETS_DEVELOPMENT_COMPONENT}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-gadgets-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cudaq-gadgets-config-version.cmake"
  DESTINATION "${CUDAQ_GADGETS_CMAKE_INSTALL_DIR}"
  COMPONENT "${CUDAQ_GADGETS_DEVELOPMENT_COMPONENT}")

if(CUDAQ_GADGETS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
