# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
set(LIBRARY_NAME cudaq-common)

set(COMMON_EXTRA_DEPS "")
set(COMMON_RUNTIME_SRC
  CustomOp.cpp
  Environment.cpp
  Executor.cpp
  Future.cpp
  Logger.cpp
  MeasureCounts.cpp
  NoiseModel.cpp
  RecordLogParser.cpp
  Resources.cpp
  ServerHelper.cpp
  Trace.cpp
)

# Create the cudaq-common library
add_library(${LIBRARY_NAME} SHARED ${COMMON_RUNTIME_SRC})

# Add into runtime_libs property
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

# Setup the includes
target_include_directories(${LIBRARY_NAME}
    PRIVATE .
        $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
        ${CMAKE_SOURCE_DIR}/tpls/eigen
        ${CMAKE_SOURCE_DIR}/runtime)

# Link privately to all dependencies
target_link_libraries(${LIBRARY_NAME} PUBLIC cudaq-operator PRIVATE spdlog::spdlog)

# Bug in GCC 12 leads to spurious warnings (-Wrestrict)
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
if (CMAKE_COMPILER_IS_GNUCXX
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
  target_compile_options(${LIBRARY_NAME} PUBLIC --param=evrp-mode=legacy)
endif()

# We can only build the RestClient support if we have OpenSSL
if(OPENSSL_FOUND)
  message(STATUS "OpenSSL Found, building REST Client.")

  target_sources(${LIBRARY_NAME} PRIVATE RestClient.cpp)
  target_link_libraries(${LIBRARY_NAME} PRIVATE cpr::cpr -Wl,--start-group ZLIB::ZLIB)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE -DCUDAQ_RESTCLIENT_AVAILABLE)
endif()


# Add a flag to indicate this is a debug build
string(TOUPPER "${CMAKE_BUILD_TYPE}" CUDAQ_UPPERCASE_CMAKE_BUILD_TYPE)
if (CUDAQ_UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  target_compile_definitions(${LIBRARY_NAME} PUBLIC CUDAQ_DEBUG)
endif()

# Install the target
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-common-targets DESTINATION lib)
install(EXPORT cudaq-common-targets
        FILE CUDAQCommonTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
if(EXISTS "$ENV{CURL_INSTALL_PREFIX}/cacert.pem")
  # Copy to build directory so that developers can run from the build directory
  file(COPY "$ENV{CURL_INSTALL_PREFIX}/cacert.pem" DESTINATION "${CMAKE_BINARY_DIR}")
  # Also include it as an install target
  install(FILES "$ENV{CURL_INSTALL_PREFIX}/cacert.pem" DESTINATION .)
endif()

## ----- MLIR-dependent core components -----
set(LIBRARY_NAME cudaq-mlir-core)
add_library(${LIBRARY_NAME} SHARED
  RuntimeMLIR.cpp
  JIT.cpp
)
target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    cudaq-common
    CCDialect
    QuakeDialect    
    OptTransforms
    OptCodeGen
    MLIRParser
    MLIRIR
    MLIRExecutionEngine
    MLIRTranslateLib
    MLIRSupport 
    MLIROptLib
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    fmt::fmt-header-only
)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-targets DESTINATION lib)

## ----- MLIR-dependent all compiler components including argument converter -----
set(LIBRARY_NAME cudaq-mlir-compiler)
add_library(${LIBRARY_NAME} SHARED
  ArgumentConversion.cpp
)
target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/tpls/spdlog/include
    ${CMAKE_SOURCE_DIR}/tpls/fmt/include
)
target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-mlir-core
  PRIVATE
    cudaq
    fmt::fmt-header-only
    MLIRPass
)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-targets DESTINATION lib)

## ----- Runtime MLIR Library -----
## We need support in the runtime for compiler-level things,
## but we have to collate this functionality in one place to avoid
## LLVM starting up static Options more than once. Create a library
## here called cudaq-mlir-runtime that cudaq-builder and cudaq-rest-qpu
## can link to and get MLIR/LLVM things uniformly

add_library(cudaq-mlir-runtime
  SHARED
    Environment.cpp
    Logger.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-mlir-runtime)

set_source_files_properties(
    Environment.cpp
    Logger.cpp
)

target_include_directories(cudaq-mlir-runtime
  PRIVATE .
    ${CMAKE_SOURCE_DIR}/tpls/spdlog/include
    ${CMAKE_SOURCE_DIR}/tpls/fmt/include
    ${CMAKE_SOURCE_DIR}/runtime)

target_link_libraries(cudaq-mlir-runtime
  PUBLIC
    cudaq-mlir-compiler
    MLIRPass
    MLIRExecutionEngine
  PRIVATE
    cudaq-common
    cudaq-operator
    fmt::fmt-header-only
    spdlog::spdlog
)

install(TARGETS cudaq-mlir-runtime EXPORT cudaq-targets DESTINATION lib)

## ----- Data layout extraction Library -----
## This library is used by runtime to get the LLVM data layout for a specific 
## kernel. This will be used by the `run` API and hence a dependency for the
## `cudaq` library.

add_library(cudaq-layout-extractor
  SHARED
    LayoutExtractor.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-layout-extractor)

target_include_directories(cudaq-layout-extractor
  PRIVATE .
    ${CMAKE_SOURCE_DIR}/runtime)

target_link_libraries(cudaq-layout-extractor
  PRIVATE
    cudaq-mlir-core
)
install(TARGETS cudaq-layout-extractor EXPORT cudaq-layout-extractor-targets DESTINATION lib)

install(EXPORT cudaq-layout-extractor-targets
        FILE CUDAQLayoutExtractorTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
