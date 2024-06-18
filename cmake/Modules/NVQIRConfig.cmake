# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(NVQIR_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(CMakeFindDependencyMacro)

get_filename_component(PARENT_DIRECTORY ${NVQIR_CMAKE_DIR} DIRECTORY)

find_dependency(CUDAQSpin REQUIRED HINTS "${PARENT_DIRECTORY}/cudaq")
find_dependency(CUDAQCommon REQUIRED HINTS "${PARENT_DIRECTORY}/cudaq")
find_dependency(fmt REQUIRED HINTS "${PARENT_DIRECTORY}/cudaq")

if(NOT TARGET nvqir::nvqir)
    include("${NVQIR_CMAKE_DIR}/NVQIRTargets.cmake")
endif()

get_filename_component(PARENT_DIRECTORY ${PARENT_DIRECTORY} DIRECTORY)
set(NVQIR_SIMULATOR_PLUGIN_PATH "${PARENT_DIRECTORY}")
get_filename_component(PARENT_DIRECTORY ${PARENT_DIRECTORY} DIRECTORY)
set(NVQIR_SIMULATOR_CONFIG_PATH "${PARENT_DIRECTORY}/targets")

function(nvqir_add_backend BackendName GPURequirements)
  set(LIBRARY_NAME nvqir-${BackendName})
  set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

  add_library(${LIBRARY_NAME} SHARED ${ARGN})
  target_link_libraries(${LIBRARY_NAME} PRIVATE nvqir::nvqir)
  install(TARGETS ${LIBRARY_NAME} DESTINATION ${NVQIR_SIMULATOR_PLUGIN_PATH})
  file (WRITE ${CMAKE_BINARY_DIR}/${BackendName}.config "NVQIR_SIMULATION_BACKEND=${BackendName}\nGPU_REQUIREMENTS=\"${GPURequirements}\"\n")
  install(FILES ${CMAKE_BINARY_DIR}/${BackendName}.config DESTINATION ${NVQIR_SIMULATOR_CONFIG_PATH})
endfunction()
