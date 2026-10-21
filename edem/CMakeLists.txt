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
  project(cudaq-edem VERSION ${CUDAQ_EDEM_VERSION} LANGUAGES C CXX)
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  # Match CUDA-Q's default library directory while honoring explicit overrides.
  if(NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Object code libraries")
  endif()
elseif(NOT DEFINED BUILD_SHARED_LIBS)
  # This default is local to this directory and its children.
  set(BUILD_SHARED_LIBS ON)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
enable_language(CXX)

option(CUDAQ_EDEM_BUILD_TESTS "Build EDEM component and packaging tests"
  ${CUDAQ_EDEM_STANDALONE_BUILD})
option(CUDAQ_EDEM_BUILD_PYTHON "Build EDEM Python bindings" OFF)

if(NOT TARGET cudaq-gf2)
  add_subdirectory(../math/gf2 "${CMAKE_CURRENT_BINARY_DIR}/gf2")
endif()

add_library(cudaq-edem lib/edem.cpp lib/composition.cpp)
add_library(cudaq::edem ALIAS cudaq-edem)
target_sources(cudaq-edem PUBLIC
  FILE_SET HEADERS
  BASE_DIRS include
  FILES include/cudaq/edem/core.h include/cudaq/edem/format.h
    include/cudaq/edem/owned.hpp
    include/cudaq/edem/python.h include/cudaq/edem/composition.h
    include/cudaq/edem/composition.hpp
)
target_link_libraries(cudaq-edem PUBLIC cudaq::gf2)
get_target_property(CUDAQ_EDEM_GF2_SOURCE_DIR cudaq-gf2 SOURCE_DIR)
target_include_directories(cudaq-edem PRIVATE "${CUDAQ_EDEM_GF2_SOURCE_DIR}/lib")
target_compile_features(cudaq-edem PUBLIC c_std_11)
target_compile_features(cudaq-edem PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cudaq-edem PRIVATE -fno-exceptions -fno-rtti)
endif()
set_target_properties(cudaq-edem PROPERTIES
  EXPORT_NAME edem
  C_EXTENSIONS OFF
  CXX_EXTENSIONS OFF
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LINKER_LANGUAGE C
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
  target_link_options(cudaq-edem PRIVATE LINKER:-dead_strip_dylibs)
  target_link_options(cudaq-gf2 PRIVATE LINKER:-dead_strip_dylibs)
elseif(UNIX)
  set_target_properties(cudaq-edem cudaq-gf2 PROPERTIES
    INSTALL_RPATH "$ORIGIN")
  target_link_options(cudaq-edem PRIVATE LINKER:--as-needed)
  target_link_options(cudaq-gf2 PRIVATE LINKER:--as-needed)
endif()

# Exact C export lists also carry ELF ABI versions. macOS uses the same lists
# without ELF version nodes. Keep them independent of compiler visibility.
foreach(library IN ITEMS cudaq-edem cudaq-gf2)
  get_target_property(library_source "${library}" SOURCE_DIR)
  set(export_map "${library_source}/lib/${library}.map")
  set_property(TARGET "${library}" PROPERTY CUDAQ_EXPORT_MAP "${export_map}")
  set_property(TARGET "${library}" APPEND PROPERTY LINK_DEPENDS "${export_map}")
  get_target_property(library_type "${library}" TYPE)
  if(library_type STREQUAL "SHARED_LIBRARY")
    if(APPLE)
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${export_map}")
      file(STRINGS "${export_map}" public_symbols REGEX "^[ \t]+cudaq_[a-z0-9_]+;")
      set(exports "")
      foreach(symbol IN LISTS public_symbols)
        string(STRIP "${symbol}" symbol)
        string(REPLACE ";" "" symbol "${symbol}")
        string(APPEND exports "_${symbol}\n")
      endforeach()
      set(export_list "${CMAKE_CURRENT_BINARY_DIR}/${library}.exports")
      file(WRITE "${export_list}" "${exports}")
      target_link_options("${library}" PRIVATE "LINKER:-exported_symbols_list,${export_list}")
    elseif(UNIX)
      target_link_options("${library}" PRIVATE "LINKER:--version-script,${export_map}")
    endif()
  endif()
endforeach()

if(CMAKE_NM)
  foreach(library IN ITEMS cudaq-edem cudaq-gf2)
    add_custom_command(TARGET cudaq-edem POST_BUILD
      COMMAND "${CMAKE_COMMAND}" "-DNM=${CMAKE_NM}"
        "-DOTOOL=${CMAKE_OTOOL}" "-DREADELF=${CMAKE_READELF}"
        "-DSHARED=$<STREQUAL:$<TARGET_PROPERTY:${library},TYPE>,SHARED_LIBRARY>"
        "-DEDEM_LIBRARY=$<TARGET_FILE:${library}>"
        "-DEDEM_EXPORT_MAP=$<TARGET_PROPERTY:${library},CUDAQ_EXPORT_MAP>"
        -P "${CUDAQ_EDEM_SOURCE_DIR}/test/check_allocators.cmake"
      VERBATIM)
    set_property(TARGET cudaq-edem APPEND PROPERTY LINK_DEPENDS
      "${CUDAQ_EDEM_SOURCE_DIR}/test/check_allocators.cmake")
  endforeach()
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

if(CUDAQ_EDEM_BUILD_PYTHON)
  enable_language(CXX)
  add_subdirectory(python)
endif()

if(CUDAQ_EDEM_BUILD_TESTS)
  enable_language(CXX)
  enable_testing()
  add_subdirectory(test)
endif()
