# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ctad-maybe-unsupported")
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP)

macro (AddQppBackend LIBRARY_NAME SOURCE_FILE)
  add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILE})

  set (QPP_DEPENDENCIES "")
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP Found. Adding build flags to QPP Backend: ${OpenMP_CXX_FLAGS}.")
    list(APPEND QPP_DEPENDENCIES OpenMP::OpenMP_CXX)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE -DHAS_OPENMP=1)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDAQ_HAS_OPENMP)
  endif()

  target_include_directories(${LIBRARY_NAME}
                 PUBLIC . .. 
                 ${CMAKE_SOURCE_DIR}/runtime/common 
                 ${CMAKE_SOURCE_DIR}/tpls/eigen 
                 ${CMAKE_SOURCE_DIR}/tpls/qpp/include)

  if(APPLE)
    set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")
  else()
    set_target_properties(${LIBRARY_NAME}
      PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${LLVM_BINARY_DIR}/lib")
  endif()

  target_link_libraries(${LIBRARY_NAME} PUBLIC ${QPP_DEPENDENCIES} PRIVATE 
                 fmt::fmt-header-only 
                 cudaq-common)
  

  install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
endmacro()



AddQppBackend(nvqir-qpp QppCircuitSimulator.cpp)
AddQppBackend(nvqir-dm QppDMCircuitSimulator.cpp)

add_target_config(density-matrix-cpu)
