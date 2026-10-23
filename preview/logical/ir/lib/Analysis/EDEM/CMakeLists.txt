# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #

qlx_select_mlir_link_libraries(_qlx_edem_mlir_libraries MLIRIR MLIRSupport)
add_mlir_library(MLIRGadgetEDEM
  CodePresentation.cpp
  IdealOutcomes.cpp
  EdemFaults.cpp
  GadgetEDEM.cpp
  DEPENDS MLIRFabricIncGen
  LINK_LIBS PUBLIC
    MLIRFabricTranslate
    cudaq::edem
    ${_qlx_edem_mlir_libraries}
)
qlx_propagate_llvm_rtti_eh(MLIRGadgetEDEM)
