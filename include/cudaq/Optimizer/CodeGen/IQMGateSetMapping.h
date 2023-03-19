/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace cudaq::iqm {

inline mlir::Value createFloatConstant(mlir::PatternRewriter &builder,
                                       mlir::Location loc, mlir::FloatType ty,
                                       double val) {
  return builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(val),
                                                      ty);
}

inline mlir::Value createTripleR(mlir::PatternRewriter &builder,
                                 mlir::Location loc, mlir::UnitAttr adj,
                                 mlir::FloatType ty, mlir::ValueRange ctrl,
                                 mlir::ValueRange targ,
                                 mlir::DenseBoolArrayAttr neg,
                                 mlir::ValueRange middle) {
  auto halfPi = createFloatConstant(builder, loc, ty, M_PI_2);
  auto zero = createFloatConstant(builder, loc, ty, 0.0);
  auto negHalfPi = builder.create<mlir::arith::NegFOp>(loc, halfPi);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{halfPi, zero}, ctrl, targ, neg);
  builder.create<quake::PhasedRxOp>(loc, adj, middle, ctrl, targ, neg);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{negHalfPi, zero}, ctrl, targ, neg);
  return {};
}

inline mlir::Value createRZ(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::ValueRange ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  auto halfPi = createFloatConstant(
      builder, loc, theta.getType().cast<mlir::FloatType>(), M_PI_2);
  mlir::Value negTheta = builder.create<mlir::arith::NegFOp>(loc, theta);
  return createTripleR(builder, loc, adj,
                       theta.getType().cast<mlir::FloatType>(), ctrl, targ, neg,
                       mlir::ArrayRef<mlir::Value>{negTheta, halfPi});
}

inline mlir::Value createR1(mlir::PatternRewriter &builder, mlir::Location loc,
                            mlir::UnitAttr adj, mlir::Value theta,
                            mlir::ValueRange ctrl, mlir::ValueRange targ,
                            mlir::DenseBoolArrayAttr neg) {
  return createRZ(builder, loc, adj, theta, ctrl, targ, neg);
}

inline mlir::Value createH(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto halfPi = createFloatConstant(builder, loc, dubTy, M_PI_2);
  auto pi = createFloatConstant(builder, loc, dubTy, M_PI);
  auto zero = createFloatConstant(builder, loc, dubTy, 0.0);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{halfPi, halfPi}, ctrl, targ, neg);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{pi, zero}, ctrl, targ, neg);
  return {};
}

inline mlir::Value createX(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto pi = createFloatConstant(builder, loc, dubTy, M_PI);
  auto zero = createFloatConstant(builder, loc, dubTy, 0.0);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{pi, zero}, ctrl, targ, neg);
  return {};
}

inline mlir::Value createY(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto halfPi = createFloatConstant(builder, loc, dubTy, M_PI_2);
  auto pi = createFloatConstant(builder, loc, dubTy, M_PI);
  builder.create<quake::PhasedRxOp>(
      loc, adj, mlir::ArrayRef<mlir::Value>{pi, halfPi}, ctrl, targ, neg);
  return {};
}

inline mlir::Value createZ(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto halfPi = createFloatConstant(builder, loc, dubTy, M_PI_2);
  auto pi = createFloatConstant(builder, loc, dubTy, M_PI);
  auto negPi = builder.create<mlir::arith::NegFOp>(loc, pi);
  return createTripleR(builder, loc, adj, dubTy, ctrl, targ, neg,
                       mlir::ArrayRef<mlir::Value>{negPi, halfPi});
}

inline mlir::Value createS(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto halfPi = createFloatConstant(builder, loc, dubTy, M_PI_2);
  auto negHalfPi = builder.create<mlir::arith::NegFOp>(loc, halfPi);
  return createTripleR(builder, loc, adj, dubTy, ctrl, targ, neg,
                       mlir::ArrayRef<mlir::Value>{negHalfPi, halfPi});
}

inline mlir::Value createT(mlir::PatternRewriter &builder, mlir::Location loc,
                           mlir::UnitAttr adj, mlir::ValueRange parm,
                           mlir::ValueRange ctrl, mlir::ValueRange targ,
                           mlir::DenseBoolArrayAttr neg) {
  auto dubTy = mlir::Float64Type::get(builder.getContext());
  auto halfPi = createFloatConstant(builder, loc, dubTy, M_PI_2);
  auto quarterPi = createFloatConstant(builder, loc, dubTy, M_PI_4);
  auto negQuarterPi = builder.create<mlir::arith::NegFOp>(loc, quarterPi);
  return createTripleR(builder, loc, adj, dubTy, ctrl, targ, neg,
                       mlir::ArrayRef<mlir::Value>{negQuarterPi, halfPi});
}

} // namespace cudaq::iqm

namespace {
#include "cudaq/Optimizer/CodeGen/IQMGateSetMapping.inc"
}
