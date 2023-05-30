/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/Builders.h"
#include <gtest/gtest.h>

using namespace mlir;

TEST(Quake, HermitianTrait) {
  MLIRContext context;
  context.loadDialect<quake::QuakeDialect>();
  OpBuilder builder(&context);

  Value qubit = builder.create<quake::AllocaOp>(builder.getUnknownLoc());
  Operation *op = builder.create<quake::HOp>(builder.getUnknownLoc(), qubit);
  ASSERT_TRUE(op->hasTrait<cudaq::Hermitian>());

  auto optor = dyn_cast<quake::OperatorInterface>(op);
  ASSERT_TRUE(optor->hasTrait<cudaq::Hermitian>());
  // The following does not work because of an MLIR bug
  // ASSERT_TRUE(optor.hasTrait<cudaq::Hermitian>());

  op = builder.create<quake::TOp>(builder.getUnknownLoc(), qubit);
  ASSERT_FALSE(op->hasTrait<cudaq::Hermitian>());

  optor = dyn_cast<quake::OperatorInterface>(op);
  ASSERT_FALSE(optor->hasTrait<cudaq::Hermitian>());
  // The following does not work because of an MLIR bug
  // ASSERT_TRUE(optor.hasTrait<cudaq::Hermitian>());
}
