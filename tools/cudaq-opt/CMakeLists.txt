# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_llvm_executable(cudaq-opt cudaq-opt.cpp)

llvm_update_compile_flags(cudaq-opt)
target_link_libraries(cudaq-opt
  PUBLIC
  ${dialect_libs}
  ${conversion_libs}

  MLIRLLVMDialect
  MLIROptLib
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect

  CCDialect
  QuakeDialect
  OptCodeGen
  OptTransforms
  CUDAQSupport
)

export_executable_symbols_for_plugins(cudaq-opt)
mlir_check_all_link_libraries(cudaq-opt)

install(TARGETS cudaq-opt DESTINATION bin)
