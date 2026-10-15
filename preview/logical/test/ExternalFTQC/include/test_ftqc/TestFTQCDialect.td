/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef TEST_FTQC_DIALECT_TD
#define TEST_FTQC_DIALECT_TD

include "mlir/IR/OpBase.td"

def TestFTQC_Dialect : Dialect {
  let name = "test_ftqc";
  let cppNamespace = "::test_ftqc";
  let summary = "out-of-tree-style CUDA-Q Logical physical extension test dialect";
}

#endif // TEST_FTQC_DIALECT_TD
