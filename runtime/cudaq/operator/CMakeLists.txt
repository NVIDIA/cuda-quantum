# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if(NOT CUDENSITYMAT_ROOT)
  SET(CUDENSITYMAT_ROOT "$ENV{CUQUANTUM_INSTALL_PREFIX}")
endif()

set(LIBRARY_NAME cudaq-operators)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

# FIXME: add boost_integrator.cpp rungakutta_integrator.cpp back to fix compile errors
set(CUDAQ_OPS_SRC
  scalar_operator.cpp elementary_operator.cpp product_operator.cpp operator_sum.cpp operator_helpers.cpp matrix_arithmetics.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_OPS_SRC})
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<BUILD_INTERFACE:/usr/local/cuda-11.8/targets/x86_64-linux/include>
       ${CUDENSITYMAT_ROOT}/include
       ${CUDAToolkit_INCLUDE_DIRS}
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

set(OPERATOR_DEPENDENCIES "")
list(APPEND OPERATOR_DEPENDENCIES fmt::fmt-header-only)
add_openmp_configurations(${LIBRARY_NAME} OPERATOR_DEPENDENCIES)

find_library(CUDENSITYMAT_LIB REQUIRED
  NAMES  cudensitymat
  HINTS   
    ${CUDENSITYMAT_ROOT}/lib
)

get_filename_component(CUDENSITYMAT_LIB_DIR ${CUDENSITYMAT_LIB} DIRECTORY)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${CUDENSITYMAT_LIB_DIR}")

target_link_libraries(${LIBRARY_NAME} PRIVATE ${OPERATOR_DEPENDENCIES})
target_link_libraries(${LIBRARY_NAME} PRIVATE ${CUDENSITYMAT_LIB})

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-operator-targets DESTINATION lib)

install(EXPORT cudaq-operator-targets
        FILE CUDAQSpinTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
