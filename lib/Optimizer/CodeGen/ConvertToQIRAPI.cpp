/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

#define DEBUG_TYPE "convert-to-qir-api"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKETOQIRAPI
#define GEN_PASS_DEF_QUAKETOQIRAPIPREP
#define GEN_PASS_DEF_QUAKETOQIRAPIFINAL
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

static std::string getGateFunctionPrefix(Operation *op) {
  return cudaq::opt::QIRQISPrefix + op->getName().stripDialect().str();
}

/// Use modifier class classes to specialize the QIR API to a particular flavor
/// of QIR. For example, the names of the actual functions in "full QIR" are
/// different than the names used by the other API flavors.
namespace {

/// The modifier class for the "full QIR" API.
struct FullQIR {
  template <typename QuakeOp>
  std::string quakeToFuncName(QuakeOp op) {
    if (op.getIsAdj())
      return getGateFunctionPrefix(op.getOperation()) + "__adj";
    return getGateFunctionPrefix(op);
  }
};

/// The base modifier class for the "profile QIR" APIs.
struct AnyProfileQIR {
  template <typename QuakeOp>
  std::string quakeToFuncName(QuakeOp op) {
    auto prefix = getGateFunctionPrefix(op.getOperation());
    if (op.getIsAdj())
      return getGateFunctionPrefix(op) + "__adj";
    return prefix + "__body";
  }
};
struct BaseProfileQIR : public AnyProfileQIR {};
struct AdaptiveProfileQIR : public AnyProfileQIR {};

struct QuakeToQIRAPIPass
    : public cudaq::opt::impl::QuakeToQIRAPIBase<QuakeToQIRAPIPass> {
  using QuakeToQIRAPIBase::QuakeToQIRAPIBase;

  template <typename A>
  void processOperation() {
    auto *op = getOperation();
  }

  void runOnOperation() override {
    if (api == "full")
      processOperation<FullQIR>();
    else if (api == "base-profile")
      processOperation<BaseProfileQIR>();
    else if (api == "adaptive-profile")
      processOperation<AdaptiveProfileQIR>();
    else
      signalPassFailure();
  }
};

struct QuakeToQIRAPIPrepPass
    : public cudaq::opt::impl::QuakeToQIRAPIPrepBase<QuakeToQIRAPIPrepPass> {
  using QuakeToQIRAPIPrepBase::QuakeToQIRAPIPrepBase;

  void runOnOperation() override {}
};

struct QuakeToQIRAPIFinalPass
    : public cudaq::opt::impl::QuakeToQIRAPIFinalBase<QuakeToQIRAPIFinalPass> {
  using QuakeToQIRAPIFinalBase::QuakeToQIRAPIFinalBase;

  void runOnOperation() override {}
};

} // namespace

void cudaq::opt::addConvertToQIRAPIPipeline(OpPassManager &pm, StringRef api) {
  QuakeToQIRAPIOptions apiOpt{.api = api.str()};
  pm.addPass(cudaq::opt::createQuakeToQIRAPIPrep());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeToQIRAPI(apiOpt));
  pm.addPass(cudaq::opt::createQuakeToQIRAPIFinal());
}

namespace {
struct QIRAPIPipelineOptions
    : public PassPipelineOptions<QIRAPIPipelineOptions> {
  PassOptions::Option<std::string> api{
      *this, "api",
      llvm::cl::desc("select the profile to convert to [full, base-profile, "
                     "adaptive-profile]"),
      llvm::cl::init("full")};
};
} // namespace

void cudaq::opt::registerToQIRAPIPipeline() {
  PassPipelineRegistration<QIRAPIPipelineOptions>(
      "convert-to-qir-api", "Convert quake to one of the QIR APIs.",
      [](OpPassManager &pm, const QIRAPIPipelineOptions &opt) {
        addConvertToQIRAPIPipeline(pm, opt.api);
      });
}
