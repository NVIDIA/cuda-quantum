# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)

# Header-only target: public headers under include/cudaq_internal/compiler/.
# Link this when you need declarations only (no cudaq-mlir-runtime .so).
add_library(cudaq-mlir-runtime-headers INTERFACE)
target_include_directories(cudaq-mlir-runtime-headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# MLIR/LLVM runtime support shared with cudaq-builder and remote platforms.
# Single library so LLVM static Options are initialized at most once.

add_library(cudaq-mlir-runtime
  SHARED
    ArgumentConversion.cpp
    Compiler.cpp
    CompiledModuleHelper.cpp
    JITTargetPipeline.cpp
    JIT.cpp
    ResourceCount.cpp
    RuntimeMLIR.cpp
    LayoutInfo.cpp
    TracePassInstrumentation.cpp
)

set_source_files_properties(
    TracePassInstrumentation.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti -fno-exceptions"
)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS cudaq-mlir-runtime)

target_link_libraries(cudaq-mlir-runtime
  PUBLIC
    cudaq-mlir-runtime-headers
  PRIVATE
    cudaq-common
    cudaq-qir-verifier
    CUDAQTargetConfig
    cudaq-logger
    fmt::fmt-header-only
    cudaqMLIR
)

install(TARGETS cudaq-mlir-runtime-headers
        EXPORT cudaq-mlir-runtime-targets
        COMPONENT Development)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include
        COMPONENT Development)

install(TARGETS cudaq-mlir-runtime
        EXPORT cudaq-mlir-runtime-targets
        DESTINATION lib
        COMPONENT Runtime)

install(EXPORT cudaq-mlir-runtime-targets
        FILE CUDAQMlirRuntimeTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
