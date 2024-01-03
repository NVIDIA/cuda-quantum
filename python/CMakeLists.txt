# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

## TODO: Clean-up following from 'old' CMakeLists.txt
add_subdirectory(utils)

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib/" CACHE INTERNAL "")
set(INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib/" CACHE INTERNAL "")

# # set(ORIGINAL_VALUE ${BUILD_SHARED_LIBS})
# SET(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

include(AddMLIRPython)

# [RFC]:
# Check how to solve this better than just disable the warning for the whole directory.
# If this is better addressed after updating to a newer LLVM version, track as an issue on GitHub.
add_compile_options(-Wno-ambiguous-reversed-operator)

# Specifies that all MLIR packages are co-located under the `mlir_cudaq`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_cudaq.")

if (CMAKE_COMPILER_IS_GNUCXX)
  # Deals with warnings in gcc12 - may not be needed for gcc 13.
  # To enable this also for clang, make sure to install the 
  # corresponding llvm-*-dev package (no change needed for the LLVM build).
  add_definitions(-flto=auto)
endif()

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(CUDAQuantumPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/QuakeOps.td
  SOURCES
    dialects/quake.py
  DIALECT_NAME quake)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT CUDAQuantumPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_cudaq"
  TD_FILE dialects/CCOps.td
  SOURCES
    dialects/cc.py
  DIALECT_NAME cc)

declare_mlir_python_extension(CUDAQuantumPythonSources.Extension
  MODULE_NAME _quakeDialects
  ADD_TO_PARENT CUDAQuantumPythonSources
  SOURCES
   CUDAQuantumExtension.cpp
   runtime/mlir/py_register_dialects.cpp
   ../runtime/cudaq/platform/default/rest/PyRemoteRESTQPU.cpp
   runtime/cudaq/qis/py_qubit_qis.cpp
   runtime/cudaq/kernels/py_common_kernels.cpp
   runtime/cudaq/platform/py_alt_launch_kernel.cpp
   runtime/cudaq/qis/py_execution_manager.cpp 
   runtime/common/py_ExecutionContext.cpp
   runtime/cudaq/algorithms/py_sample_async.cpp
   runtime/cudaq/algorithms/py_observe_async.cpp
   runtime/cudaq/platform/JITExecutionCache.cpp
   utils/LinkedLibraryHolder.cpp
   runtime/cudaq/target/py_runtime_target.cpp
   runtime/cudaq/target/py_testing_utils.cpp
   runtime/common/py_SampleResult.cpp
   runtime/common/py_ObserveResult.cpp
   runtime/common/py_NoiseModel.cpp
   runtime/cudaq/spin/py_matrix.cpp
   runtime/cudaq/spin/py_spin_op.cpp 
   runtime/cudaq/algorithms/py_optimizer.cpp 
   ../runtime/cudaq/platform/common/QuantumExecutionQueue.cpp

  EMBED_CAPI_LINK_LIBS
   CUDAQuantumMLIRCAPI
   MLIRCAPIExecutionEngine
  PRIVATE_LINK_LIBS
   OptCodeGen
   OptTransforms
)

target_include_directories(CUDAQuantumPythonSources.Extension INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/python 
    ${CMAKE_SOURCE_DIR}/python/utils)
target_link_libraries(CUDAQuantumPythonSources.Extension INTERFACE 
    cudaq 
    cudaq-common 
    cudaq-em-default
    fmt::fmt-header-only
    )

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(CUDAQuantumPythonCAPI
  INSTALL_COMPONENT CUDAQuantumPythonModules
  INSTALL_DESTINATION mlir_cudaq/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/mlir_cudaq/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(CUDAQuantumPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/mlir_cudaq"
  INSTALL_PREFIX "mlir_cudaq"
  DECLARED_SOURCES
    CUDAQuantumPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    CUDAQuantumPythonCAPI
  )

file (COPY cudaq DESTINATION ${MLIR_BINARY_DIR}/)

# set(BUILD_SHARED_LIBS ${ORIGINAL_VALUE})
set_target_properties(CUDAQuantumPythonModules CUDAQuantumPythonCAPI CUDAQuantumPythonSources.Extension
                      PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../../lib/")


## FIXME: Delete '_pycudaq' target?
set(LIBRARY_NAME _pycudaq)
if (NOT SKBUILD)
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../../lib:$ORIGIN/../../lib/plugins")
else()
  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN/../cuda_quantum.libs")
endif()
pybind11_add_module(_pycudaq 
                     cudaq/_cudaq.cpp 
                     runtime/cudaq/target/py_runtime_target.cpp 
                     runtime/cudaq/target/py_testing_utils.cpp
                     runtime/cudaq/builder/py_kernel_builder.cpp
                     runtime/cudaq/builder/py_QuakeValue.cpp
                     runtime/cudaq/algorithms/py_observe.cpp
                     runtime/cudaq/algorithms/py_sample.cpp
                     runtime/cudaq/algorithms/py_optimizer.cpp
                     runtime/cudaq/algorithms/py_state.cpp
                     runtime/cudaq/algorithms/py_vqe.cpp
                     runtime/cudaq/spin/py_matrix.cpp
                     runtime/cudaq/spin/py_spin_op.cpp
                     runtime/cudaq/kernels/py_chemistry.cpp
                     runtime/common/py_ObserveResult.cpp
                     runtime/common/py_SampleResult.cpp                     
                     runtime/common/py_NoiseModel.cpp)

target_compile_options(_pycudaq PRIVATE -Wno-attributes)                     
target_include_directories(_pycudaq PRIVATE .)
target_link_libraries(_pycudaq 
    PRIVATE 
      nvqir
      cudaq 
      cudaq-builder 
      cudaq-em-default 
      cudaq-platform-default 
      cudaq-py-utils)

if (NOT SKBUILD)
  install(DIRECTORY cudaq DESTINATION .)
  install(TARGETS _pycudaq DESTINATION cudaq)
  # Also move these to the build directory for testing.
  file (COPY cudaq DESTINATION ${CMAKE_BINARY_DIR}/)
  add_custom_command(TARGET _pycudaq POST_BUILD 
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_pycudaq> 
              ${CMAKE_BINARY_DIR}/cudaq)
else()
  install(TARGETS _pycudaq LIBRARY DESTINATION cudaq)
endif()

add_subdirectory(runtime/cudaq/distributed)
add_subdirectory(runtime/cudaq/domains/plugins)

# Do we have pytest and numpy? 
execute_process(COMMAND ${Python_EXECUTABLE} -m pytest --version
  OUTPUT_VARIABLE PYTEST_output
  ERROR_VARIABLE  PYTEST_error
  RESULT_VARIABLE PYTEST_result)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import numpy"
  OUTPUT_VARIABLE NUMPY_output
  ERROR_VARIABLE NUMPY_error
  RESULT_VARIABLE NUMPY_result)

if(CUDAQ_BUILD_TESTS)
  if(${PYTEST_result} EQUAL 0 AND ${NUMPY_result} EQUAL 0)
    message(STATUS "Pytest and Numpy found, building Python tests.")
    add_subdirectory(tests)
  else()
    message(FATAL_ERROR "CUDA Quantum Python Warning - CUDAQ_BUILD_TESTS=TRUE but can't find numpy or pytest modules required for testing.")
  endif()
endif()


## The Python bindings module for Quake dialect depends on CUDAQ libraires 
## which it can't locate since they are in "../../lib" and the 'rapth' is set
## to '$ORIGIN' by default.

## Following has no effect; edit a POST_BUILD custom operation? But, which target to tie it to?
# set(QuakePyModule "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/mlir_cudaq/_mlir_libs/_quakeDialects.cpython-310-x86_64-linux-gnu.so")
# if(CMAKE_INSTALL_COMPONENT STREQUAL "CUDAQuantumPythonModules" OR NOT CMAKE_INSTALL_COMPONENT)
#   if(EXISTS "${QuakePyModule}")
#     file(RPATH_CHANGE
#          FILE "${QuakePyModule}"
#          OLD_RPATH "\$ORIGIN"
#          NEW_RPATH "\$ORIGIN/../../lib")
#   endif()
# endif()

## Other options: 
##  -- use patchelf as post-install step
##  -- create symbolic links from top-level 'lib' directory to 'mlir_cudaq/_mlir_libs'
