# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)
set(LIBRARY_NAME cudaq-common)

set(COMMON_EXTRA_DEPS "")
set(COMMON_RUNTIME_SRC
  Environment.cpp
  Executor.cpp
  Future.cpp
  Logger.cpp 
  MeasureCounts.cpp 
  NoiseModel.cpp 
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

## ----- Runtime MLIR Library ----- 
## We need support in the runtime for compiler-level things, 
## but we have to collate this functionality in one place to avoid 
## LLVM starting up static Options more than once. Create a library 
## here called cudaq-mlir-runtime that cudaq-builder and cudaq-rest-qpu 
## can link to and get MLIR/LLVM things uniformly

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(cudaq-mlir-runtime
  SHARED
    ArgumentConversion.cpp
    Environment.cpp
    JIT.cpp
    Logger.cpp
    RuntimeMLIR.cpp
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-mlir-runtime)

# Note: JIT.cpp contains throw statements. Should RTTI be enabled?
set_source_files_properties(
    ArgumentConversion.cpp
    Environment.cpp
    JIT.cpp
    Logger.cpp
    RuntimeMLIR.cpp
  PROPERTIES COMPILE_FLAGS -fno-rtti
)

target_include_directories(cudaq-mlir-runtime
  PRIVATE .
    ${CMAKE_SOURCE_DIR}/tpls/spdlog/include
    ${CMAKE_SOURCE_DIR}/tpls/fmt/include
    ${CMAKE_SOURCE_DIR}/runtime)

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
    MLIRLLVMToLLVMIRTranslation
  PRIVATE
    spdlog::spdlog)

install(TARGETS cudaq-mlir-runtime DESTINATION lib)
