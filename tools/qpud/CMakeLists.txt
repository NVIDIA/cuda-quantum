# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

# Create the qpud target
add_executable(qpud qpud.cpp)

# Add source for Kernel JIT compilation
add_subdirectory(jit)

# Add utility sources
add_subdirectory(utils)

# Add the backends 
add_subdirectory(backends)

# Add some include paths
target_include_directories(qpud SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/tpls/json/nlohmann)
target_include_directories(qpud PRIVATE . 
  backends/ 
  backends/rest 
  utils
  jit
  ${LLVM_INCLUDE_DIRS}
)

# Link with rpc, nvqir, cudaq and MLIR libs
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(qpud PRIVATE fmt::fmt-header-only
  rpc 
  nvqir
  ${dialect_libs}
  ${conversion_libs}
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
  
  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
  cudaq
)

# Install the target
install(TARGETS qpud DESTINATION bin)
