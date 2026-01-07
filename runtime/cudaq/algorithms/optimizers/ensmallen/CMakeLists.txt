# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-ensmallen)

set (CUDAQ_TPLS_DIR ${CMAKE_SOURCE_DIR}/tpls)

set(BLA_STATIC ON)
find_package(BLAS REQUIRED)

add_library(${LIBRARY_NAME} SHARED ensmallen.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_compile_definitions(${LIBRARY_NAME} PRIVATE -DARMA_DONT_USE_LAPACK)

set (ENSMALLEN_DEPENDENCIES "")
list(APPEND ENSMALLEN_DEPENDENCIES BLAS::BLAS)
add_openmp_configurations(${LIBRARY_NAME} ENSMALLEN_DEPENDENCIES)

if (CUDAQ_OPTIMIZER_VERBOSE) 
  target_compile_definitions(${LIBRARY_NAME} PRIVATE -DENS_PRINT_INFO)
endif()

target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime)
target_include_directories(${LIBRARY_NAME} SYSTEM
     PRIVATE ${CUDAQ_TPLS_DIR}/armadillo/include 
             ${CUDAQ_TPLS_DIR}/ensmallen/include)

target_link_libraries(${LIBRARY_NAME} PRIVATE ${ENSMALLEN_DEPENDENCIES})

if (NOT APPLE)
  # Don't export BLAS symbols to prevent symbol clashes.
  # Exclude-libs is not supported on macOs
  target_link_options(${LIBRARY_NAME} PRIVATE -Wl,--exclude-libs,ALL)
endif()

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-ensmallen-targets DESTINATION lib)
install(EXPORT cudaq-ensmallen-targets
        FILE CUDAQEnsmallenTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
