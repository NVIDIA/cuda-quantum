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

add_llvm_executable(qtx-translate qtx-translate.cpp)

llvm_update_compile_flags(qtx-translate)
target_link_libraries(qtx-translate
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRTranslateLib
  MLIRSupport

  CUDAQTargetOpenQASM
  CUDAQTargetIQM
  QTXDialect
)

mlir_check_all_link_libraries(qtx-translate)

install(TARGETS qtx-translate DESTINATION bin)
