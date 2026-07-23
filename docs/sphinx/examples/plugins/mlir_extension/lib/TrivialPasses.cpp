/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "Trivial/TrivialPasses.h"
#include "Trivial/TrivialDialect.h"
#include "Trivial/TrivialOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace trivial {
// TableGen-generated pass base class (impl::TrivialPassBase) +
// createTrivialPass.
#define GEN_PASS_DEF_TRIVIALPASS
#include "Trivial/TrivialPasses.h.inc"
} // namespace trivial

using namespace mlir;

namespace {

struct TrivialPass : public trivial::impl::TrivialPassBase<TrivialPass> {
  using TrivialPassBase::TrivialPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    trivial::NopOp::create(builder, builder.getUnknownLoc());
    module->setAttr("trivial.visited", builder.getUnitAttr());
  }
};

} // namespace
