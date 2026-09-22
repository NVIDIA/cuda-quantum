# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(EXE_NAME cudaq-target-db-gen)

add_executable(${EXE_NAME} cudaq-target-db-gen.cpp)
# Same build-tree directory as the other tools. Callers use that path
# directly (lit, plugin tests, nvq++); the install rule below matches it.
set_target_properties(${EXE_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_link_libraries(${EXE_NAME} PRIVATE
  CUDAQTargetConfigObjects reflectcpp ${CMAKE_DL_LIBS})
# If not CUDAQ_DISABLE_TOOLS, installl it so external plugin authors can compile
# their own targets through cudaq-target-db-gen --plugin. See
# `docs/sphinx/using/extending/packaging.rst`.
if (NOT CUDAQ_DISABLE_TOOLS)
  install(TARGETS ${EXE_NAME} DESTINATION bin COMPONENT Nvqpp)
endif()
