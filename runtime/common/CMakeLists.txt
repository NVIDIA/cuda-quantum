# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-common)

set(COMMON_EXTRA_DEPS "")
set(COMMON_RUNTIME_SRC
  Logger.cpp 
  MeasureCounts.cpp 
  NoiseModel.cpp 
  ServerHelper.cpp 
  Resources.cpp
  Future.cpp
)

# Create the cudaq-common library
add_library(${LIBRARY_NAME} SHARED ${COMMON_RUNTIME_SRC})

# Setup the includes
target_include_directories(${LIBRARY_NAME} 
    PRIVATE . 
        $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
        ${CMAKE_SOURCE_DIR}/tpls/eigen 
        ${CMAKE_SOURCE_DIR}/runtime)

# Link privately to all dependencies
target_link_libraries(${LIBRARY_NAME} PUBLIC cudaq-spin PRIVATE spdlog::spdlog)

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

  include(FetchContent)
  FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG 871ed52d350214a034f6ef8a3b8f51c5ce1bd400) 

  find_package(CURL COMPONENTS HTTP HTTPS QUIET)

  if (CURL_FOUND)
    message(STATUS "Curl found. Using system curl for CPR.")
    set(CPR_FORCE_USE_SYSTEM_CURL TRUE)
  else()
    message(STATUS "Curl not found. Building curl for CPR.")
    set(CPR_FORCE_USE_SYSTEM_CURL FALSE)
  endif()
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
  if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(CMAKE_BUILD_TYPE "Debug")
  endif()

  FetchContent_GetProperties(cpr)
  if(NOT cpr_POPULATED)
    FetchContent_Populate(cpr)
    add_subdirectory(${cpr_SOURCE_DIR} ${cpr_BINARY_DIR} EXCLUDE_FROM_ALL)
    target_compile_options(cpr PRIVATE "-w")
  endif()

  target_sources(${LIBRARY_NAME} PRIVATE RestClient.cpp)
  target_link_libraries(${LIBRARY_NAME} PRIVATE cpr::cpr)
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

## ----- Runtime MLIR Library ----- 
## We need support in the runtime for compiler-level things, 
## but we have to collate this functionality in one place to avoid 
## LLVM starting up static Options more than once. Create a library 
## here called cudaq-mlir-runtime that cudaq-builder and cudaq-rest-qpu 
## can link to and get MLIR/LLVM things uniformly

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(cudaq-mlir-runtime SHARED RuntimeMLIR.cpp)

target_include_directories(cudaq-mlir-runtime
  PRIVATE . ${CMAKE_SOURCE_DIR}/runtime)

target_link_libraries(cudaq-mlir-runtime 
  PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    CCDialect
    QuakeDialect
    OptCodeGen
    OptTransforms
    MLIRTranslateLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTranslateLib
    MLIRSupport
    MLIROptLib
    MLIRExecutionEngine
    MLIRTransforms
    MLIRTargetLLVMIRExport
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation)

install(TARGETS cudaq-mlir-runtime DESTINATION lib)
