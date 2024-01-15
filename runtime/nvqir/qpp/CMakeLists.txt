# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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

  set (EXTRA_LIBS "")
  if(OpenMP_CXX_FOUND)
    set(EXTRA_LIBS OpenMP::OpenMP_CXX)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE EIGEN_HAS_OPENMP)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_OPENMP)
  endif()

  target_include_directories(${LIBRARY_NAME}
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpls/eigen>
        $<INSTALL_INTERFACE:include>)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC libqpp
    PRIVATE fmt::fmt-header-only cudaq-common ${EXTRA_LIBS})

  set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")

  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()



AddQppBackend(nvqir-qpp QppCircuitSimulator.cpp)
AddQppBackend(nvqir-dm QppDMCircuitSimulator.cpp)

add_target_config(qpp-cpu)
add_target_config(density-matrix-cpu)
