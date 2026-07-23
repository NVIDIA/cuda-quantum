# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(cudaq-opt cudaq-opt.cpp)

llvm_update_compile_flags(cudaq-opt)
if(APPLE)
  # Link libcudaqMLIR dynamically to avoid duplicate symbols on macOS.
  target_link_libraries(cudaq-opt
    PUBLIC
    CUDAQSupport
    cudaqMLIR
  )
  target_link_options(cudaq-opt BEFORE PRIVATE "$<TARGET_FILE:cudaqMLIR>")
else()
  # Link MLIR statically for performance; the linker unifies duplicate symbols.
  target_link_libraries(cudaq-opt
    PUBLIC
    MLIRLLVMDialect
    MLIROptLib
    MLIRIR
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRArithDialect

    CCDialect
    QECDialect
    QuakeDialect
    OptCodeGen
    OptTransforms
    CUDAQSupport
  )
  mlir_check_all_link_libraries(cudaq-opt)
endif()

export_executable_symbols_for_plugins(cudaq-opt)

install(TARGETS cudaq-opt DESTINATION bin)
