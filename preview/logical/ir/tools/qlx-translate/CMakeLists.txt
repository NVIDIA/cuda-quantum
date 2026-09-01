# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# qlx-translate: translation between QLX MLIR and external formats (e.g. Stim).

add_llvm_executable(qlx-translate
  qlx-translate.cpp
)

target_link_libraries(qlx-translate PRIVATE
  MLIRQLXDialect
  MLIRFabricDialect
  MLIRFabricTranslate

  MLIRArithDialect
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRTranslateLib
)

llvm_update_compile_flags(qlx-translate)

# Link MLIR statically, like cudaq-translate.
cudaq_use_static_mlir(qlx-translate)

set_target_properties(qlx-translate PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${QLX_TOOLS_DIR}
)

if(SKBUILD)
  set_target_properties(qlx-translate PROPERTIES
    INSTALL_RPATH "${QLX_TOOLS_INSTALL_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE)
endif()

install(TARGETS qlx-translate
  RUNTIME DESTINATION "${QLX_TOOLS_INSTALL_DIR}"
  COMPONENT QLXTools)
