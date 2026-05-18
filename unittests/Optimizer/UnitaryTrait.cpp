/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/Builders.h"
#include <gtest/gtest.h>

using namespace mlir;

TEST(Quake, UnitaryTrait) {
  MLIRContext context;
  context.loadDialect<cudaq::quake::QuakeDialect, mlir::arith::ArithDialect>();
  OpBuilder builder(&context);

  Value qubit0 =
      cudaq::quake::AllocaOp::create(builder, builder.getUnknownLoc());
  Value qubit1 =
      cudaq::quake::AllocaOp::create(builder, builder.getUnknownLoc());

  Operation *op = cudaq::quake::ExpPauliOp::create(
      builder, builder.getUnknownLoc(), {}, {}, {qubit0, qubit1}, "ZX");
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::HOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::HOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  Value pi_2 = cudaq::opt::factory::createFloatConstant(
      builder.getUnknownLoc(), builder, M_PI_2, builder.getF64Type());
  op = cudaq::quake::R1Op::create(builder, builder.getUnknownLoc(), pi_2,
                                  qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::RxOp::create(builder, builder.getUnknownLoc(), pi_2,
                                  qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::RyOp::create(builder, builder.getUnknownLoc(), pi_2,
                                  qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::RzOp::create(builder, builder.getUnknownLoc(), pi_2,
                                  qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::SOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::SwapOp::create(builder, builder.getUnknownLoc(), qubit0,
                                    qubit1);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::TOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_FALSE(op->hasTrait<cudaq::Hermitian>());

  op = cudaq::quake::U2Op::create(builder, builder.getUnknownLoc(),
                                  {pi_2, pi_2}, qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::U3Op::create(builder, builder.getUnknownLoc(),
                                  {pi_2, pi_2, pi_2}, qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::XOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::YOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());

  op = cudaq::quake::ZOp::create(builder, builder.getUnknownLoc(), qubit0);
  ASSERT_TRUE(op->hasTrait<cudaq::Unitary>());
}
