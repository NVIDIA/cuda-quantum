# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# DISABLE_INSTALL keeps these out of MLIRTargets export set: they are
# embedded into the QLXPythonCAPI common library at build time.

set(_qlx_optional_pass_libraries)
if(QLX_USE_CUDAQ_SDK)
  list(APPEND _qlx_optional_pass_libraries MLIRQuakeToQLX)
endif()

add_mlir_public_c_api_library(QLXCAPIPasses
  Passes.cpp
  DISABLE_INSTALL

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR

  # Conversion
  MLIRQLXToLVM

  # Dialect
  MLIRFabricTransforms
  MLIRPhysTransforms
  ${_qlx_optional_pass_libraries}
)

if(QLX_USE_CUDAQ_SDK)
  target_compile_definitions(obj.QLXCAPIPasses PRIVATE
    QLX_HAS_CUDAQ_QUAKE=1)
endif()
