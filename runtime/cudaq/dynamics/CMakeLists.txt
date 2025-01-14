# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-operators)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

set(CUDAQ_OPS_SRC
  scalar_operators.cpp elementary_operators.cpp product_operators.cpp operator_sum.cpp schedule.cpp definition.cpp helpers.cpp
)

add_library(${LIBRARY_NAME} SHARED ${CUDAQ_OPS_SRC})
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
       $<INSTALL_INTERFACE:include>
    PRIVATE .)

set (OPERATOR_DEPENDENCIES "")
list(APPEND OPERATOR_DEPENDENCIES fmt::fmt-header-only)
add_openmp_configurations(${LIBRARY_NAME} OPERATOR_DEPENDENCIES)

target_link_libraries(${LIBRARY_NAME} PRIVATE ${OPERATOR_DEPENDENCIES})

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-operator-targets DESTINATION lib)

install(EXPORT cudaq-operator-targets
        FILE CUDAQSpinTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
