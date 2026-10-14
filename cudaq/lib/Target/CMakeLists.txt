# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Support for building as standalone project.
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
       CUDAQ_TARGET_STANDALONE_BUILD)
if(CUDAQ_TARGET_STANDALONE_BUILD)
  cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
  project(cudaq-target LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
  set(CUDAQ_TARGET_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
  set(CUDAQ_CORE_LIBRARY_COMPONENT Runtime)
  set(CUDAQ_CORE_DEVELOPMENT_COMPONENT Development)
  enable_testing()
else()
  set(CUDAQ_TARGET_REPO_ROOT "${CMAKE_SOURCE_DIR}")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchDependencies.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TargetDatabase.cmake)

# Acyclic bootstrap: the lightweight config library feeds the YAML parser
# objects, which feed cudaq-target-db-gen, whose generated table compiles into
# the static database. The catalog publicly re-exports the config library and
# privately links the database.
#   CUDAQTargetConfig -> CUDAQTargetYamlObjects -> cudaq-target-db-gen
#         |                                          |
#         |                                          v
#         +--------------------------> CUDAQTargetDatabase
#         |                                          |
#         v                                          v
#      CUDAQTargetCatalog <--------------------------

# Lightweight, installed library: the in-memory target configuration model.
# Dependency-free public dependency for components (e.g. cudaq-common) that
# read and modify target configurations but never load targets themselves.
add_library(CUDAQTargetConfig STATIC TargetConfig.cpp)
target_include_directories(CUDAQTargetConfig PUBLIC
  $<BUILD_INTERFACE:${CUDAQ_TARGET_REPO_ROOT}/cudaq/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(CUDAQTargetConfig PUBLIC cxx_std_20)
target_compile_options(CUDAQTargetConfig PRIVATE "-fexceptions" "-frtti")
set_target_properties(CUDAQTargetConfig PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# Internal YAML parser objects (schema v1). Used by cudaq-target-db-gen and
# embedded into the catalog; not installed, never a link interface on its own.
add_library(CUDAQTargetYamlObjects OBJECT
  Yaml/TargetConfigSchema.cpp
  Yaml/v1/TargetConfigV1.cpp
)
target_include_directories(CUDAQTargetYamlObjects PUBLIC
  $<BUILD_INTERFACE:${CUDAQ_TARGET_REPO_ROOT}/cudaq/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(CUDAQTargetYamlObjects PRIVATE CUDAQTargetConfig reflectcpp)
target_compile_options(CUDAQTargetYamlObjects PRIVATE "-fexceptions" "-frtti")
set_target_properties(CUDAQTargetYamlObjects PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# The generator to build the targets database.
add_subdirectory(tools/cudaq-target-db-gen)

# Static target database: hand-written accessors plus the generated table
# (attached by cudaq_finalize_target_database()). No link back to the config
# objects or the catalog.
add_library(CUDAQTargetDatabase STATIC TargetDatabase.cpp)
target_include_directories(CUDAQTargetDatabase
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_TARGET_REPO_ROOT}/cudaq/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(CUDAQTargetDatabase PUBLIC cxx_std_20)
set_target_properties(CUDAQTargetDatabase PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(CUDAQTargetDatabase PUBLIC CUDAQTargetConfig)

# Finalize the database here if standalone, otherwise wait until after
# external projects have registered their targets.
if(CUDAQ_TARGET_STANDALONE_BUILD)
  cudaq_finalize_target_database()
endif()

# Final full library: the catalog, embedding the database and dependencies.
add_library(CUDAQTargetCatalog
  TargetCatalog.cpp
  TargetRuntimeArgs.cpp
  TargetPluginLibrary.cpp
  $<TARGET_OBJECTS:CUDAQTargetYamlObjects>
  $<TARGET_OBJECTS:reflectcpp>
  $<TARGET_OBJECTS:yaml-cpp>)
# Ensure library lands in the top-level lib directory.
set_target_properties(CUDAQTargetCatalog PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(CUDAQTargetCatalog
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_TARGET_REPO_ROOT}/cudaq/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(CUDAQTargetCatalog PUBLIC cxx_std_20)
target_compile_options(CUDAQTargetCatalog PRIVATE "-fexceptions" "-frtti")
target_link_libraries(CUDAQTargetCatalog
  PUBLIC
    CUDAQTargetConfig
  PRIVATE
    CUDAQTargetDatabase
    ${CMAKE_DL_LIBS})

install(TARGETS CUDAQTargetConfig
        EXPORT CUDAQTargetConfigTargets
        DESTINATION lib
        COMPONENT ${CUDAQ_CORE_LIBRARY_COMPONENT})
install(EXPORT CUDAQTargetConfigTargets
        FILE CUDAQTargetConfigTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/CUDAQTargetConfig
        COMPONENT ${CUDAQ_CORE_DEVELOPMENT_COMPONENT})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDAQTargetConfigConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CUDAQTargetConfigConfig.cmake
  INSTALL_DESTINATION lib/cmake/CUDAQTargetConfig)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CUDAQTargetConfigConfig.cmake
        DESTINATION lib/cmake/CUDAQTargetConfig
        COMPONENT ${CUDAQ_CORE_DEVELOPMENT_COMPONENT})

install(TARGETS CUDAQTargetCatalog CUDAQTargetDatabase
        EXPORT CUDAQTargetCatalogTargets
        DESTINATION lib
        COMPONENT Runtime)
install(EXPORT CUDAQTargetCatalogTargets
        FILE CUDAQTargetCatalogTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/CUDAQTargetCatalog
        COMPONENT Development)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDAQTargetCatalogConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CUDAQTargetCatalogConfig.cmake
  INSTALL_DESTINATION lib/cmake/CUDAQTargetCatalog)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CUDAQTargetCatalogConfig.cmake
        DESTINATION lib/cmake/CUDAQTargetCatalog
        COMPONENT Development)

if(CUDAQ_TARGET_STANDALONE_BUILD)
  # The full build installs these headers from the tree-wide include dir.
  install(FILES
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetAbiVersion.h
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetCatalog.h
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetConfig.h
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetConfigYaml.h
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetDatabase.h
    ${CUDAQ_TARGET_REPO_ROOT}/cudaq/include/cudaq/Target/TargetPluginLibrary.h
    DESTINATION include/cudaq/Target
    COMPONENT Development)
endif()

# Component-local LIT suite.
if (CUDAQ_BUILD_TESTS)
  add_subdirectory(test)
endif()
