# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
set(TOOL_NAME "cudaq-translate")

add_llvm_executable(${TOOL_NAME} ${TOOL_NAME}.cpp)

llvm_update_compile_flags(${TOOL_NAME})
if(APPLE)
  # Link libcudaqMLIR dynamically to avoid duplicate symbols on macOS.
  target_link_libraries(${TOOL_NAME}
    PRIVATE
      cudaq-qir-verifier
      cudaqMLIR
  )
  target_link_options(${TOOL_NAME} BEFORE PRIVATE "$<TARGET_FILE:cudaqMLIR>")
else()
  # Link MLIR statically for performance; the linker unifies duplicate symbols.
  target_link_libraries(${TOOL_NAME}
    PRIVATE
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRTranslateLib
      MLIRSupport
      MLIROptLib
      MLIRExecutionEngine
      MLIRTransforms
      MLIRTargetLLVMIRExport
      MLIRLLVMCommonConversion
      MLIRBuiltinToLLVMIRTranslation
      MLIRLLVMToLLVMIRTranslation
      MLIRFuncInlinerExtension
      MLIRLLVMIRTransforms

      CCDialect
      OptCodeGen
      OptTransforms
      QECDialect
      QuakeDialect
      cudaq-qir-verifier
  )
  mlir_check_all_link_libraries(${TOOL_NAME})
endif()

install(TARGETS ${TOOL_NAME} DESTINATION bin)
