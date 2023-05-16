/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace cudaq::decomposition {

inline mlir::Value createFloatConstant(mlir::PatternRewriter &builder,
                                       mlir::Location loc, mlir::FloatType ty,
                                       double val) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(val),
                                                      ty);
}

inline std::size_t createRx(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::Value ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  builder.create<quake::SOp>(loc, adj, mlir::ValueRange{}, mlir::ValueRange{},
                             targ, neg);
  builder.create<quake::XOp>(loc, adj, mlir::ValueRange{},
                             mlir::ValueRange{ctrl}, targ, neg);
  auto two = createFloatConstant(builder, loc,
                                 theta.getType().cast<mlir::FloatType>(), 2.0);
  auto halfTheta = builder.create<mlir::arith::DivFOp>(loc, theta, two);
  auto negHalfTheta = builder.create<mlir::arith::NegFOp>(loc, halfTheta);
  builder.create<quake::RyOp>(loc, adj, mlir::ValueRange{negHalfTheta},
                              mlir::ValueRange{}, targ, neg);
  builder.create<quake::XOp>(loc, adj, mlir::ValueRange{},
                             mlir::ValueRange{ctrl}, targ, neg);
  builder.create<quake::RyOp>(loc, adj, mlir::ValueRange{halfTheta},
                              mlir::ValueRange{}, targ, neg);
  auto halfPi = createFloatConstant(
      builder, loc, theta.getType().cast<mlir::FloatType>(), M_PI_2);
  auto negHalfPi = builder.create<mlir::arith::NegFOp>(loc, halfPi);
  builder.create<quake::RzOp>(loc, adj, mlir::ValueRange{negHalfPi},
                              mlir::ValueRange{}, targ, neg);
  return 0;
}

template <typename A>
std::size_t createR_(mlir::PatternRewriter &builder, mlir::Location loc,
                     mlir::UnitAttr adj, mlir::Value theta, mlir::Value ctrl,
                     mlir::ValueRange targ, mlir::DenseBoolArrayAttr neg) {
  auto two = createFloatConstant(builder, loc,
                                 theta.getType().cast<mlir::FloatType>(), 2.0);
  auto halfTheta = builder.create<mlir::arith::DivFOp>(loc, theta, two);
  builder.create<A>(loc, adj, mlir::ValueRange{halfTheta}, mlir::ValueRange{},
                    targ, neg);
  builder.create<quake::XOp>(loc, adj, mlir::ValueRange{},
                             mlir::ValueRange{ctrl}, targ, neg);
  auto negHalfTheta = builder.create<mlir::arith::NegFOp>(loc, halfTheta);
  builder.create<A>(loc, adj, mlir::ValueRange{negHalfTheta},
                    mlir::ValueRange{}, targ, neg);
  builder.create<quake::XOp>(loc, adj, mlir::ValueRange{},
                             mlir::ValueRange{ctrl}, targ, neg);
  return 0;
}

inline std::size_t createRy(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::Value ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  return createR_<quake::RyOp>(builder, loc, adj, theta, ctrl, targ, neg);
}

inline std::size_t createRz(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::Value ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  return createR_<quake::RzOp>(builder, loc, adj, theta, ctrl, targ, neg);
}

inline std::size_t createR1(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::Value ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  builder.create<quake::RzOp>(loc, adj, mlir::ValueRange{theta},
                              mlir::ValueRange{ctrl}, targ, neg);
  auto two = createFloatConstant(builder, loc,
                                 theta.getType().cast<mlir::FloatType>(), 2.0);
  auto halfTheta = builder.create<mlir::arith::DivFOp>(loc, theta, two);
  builder.create<quake::RzOp>(loc, adj, mlir::ValueRange{halfTheta},
                              mlir::ValueRange{}, mlir::ValueRange{ctrl}, neg);
  return 0;
}

} // namespace cudaq::decomposition

namespace {
#include "cudaq/Optimizer/CodeGen/DecompositionGateSetMapping.inc"
}
