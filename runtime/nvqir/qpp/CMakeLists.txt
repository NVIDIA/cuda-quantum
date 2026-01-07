# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

macro (AddQppBackend LIBRARY_NAME SOURCE_FILE)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILE})
  set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

  set (QPP_DEPENDENCIES "")
  list(APPEND QPP_DEPENDENCIES fmt::fmt-header-only cudaq-common)
  add_openmp_configurations(${LIBRARY_NAME} QPP_DEPENDENCIES)

  target_include_directories(${LIBRARY_NAME}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
        $<INSTALL_INTERFACE:include>)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC libqpp
    PRIVATE ${QPP_DEPENDENCIES})

  set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()



AddQppBackend(nvqir-qpp QppCircuitSimulator.cpp)
AddQppBackend(nvqir-dm QppDMCircuitSimulator.cpp)

add_target_config(qpp-cpu)
add_target_config(density-matrix-cpu)
