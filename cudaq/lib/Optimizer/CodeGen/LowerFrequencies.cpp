/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERFREQUENCIES
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt
using namespace mlir;
namespace {
// Conservative prototype proof: gates and local scalar computation only.
// In particular, reject allocation/discard, measurement, reset, noise, emit,
// external calls, and writes through caller-owned classical storage.
static bool isUnitaryPreparation(func::FuncOp prep,
                                 llvm::SmallPtrSetImpl<Operation *> &active) {
  if (prep.empty() || !active.insert(prep).second)
    return false;
  for (auto type : prep.getArgumentTypes())
    if (!isa<IntegerType, FloatType, cudaq::quake::RefType,
             cudaq::quake::VeqType>(type)) {
      active.erase(prep);
      return false;
    }
  bool eligible =
      !prep.walk([&](Operation *op) {
             if (op == prep.getOperation())
               return WalkResult::advance();
             if (auto call = dyn_cast<func::CallOp>(op)) {
               auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                   call, call.getCalleeAttr());
               return callee && isUnitaryPreparation(callee, active)
                          ? WalkResult::advance()
                          : WalkResult::interrupt();
             }
             if (isa<cudaq::quake::OperatorInterface>(op) ||
                 isa<func::ReturnOp, cudaq::cc::AllocaOp, cudaq::cc::ScopeOp,
                     cudaq::cc::ContinueOp, cudaq::cc::ConditionOp,
                     cudaq::cc::IfOp, cudaq::cc::LoopOp>(op))
               return WalkResult::advance();
             if (auto load = dyn_cast<cudaq::cc::LoadOp>(op))
               return load.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>()
                          ? WalkResult::advance()
                          : WalkResult::interrupt();
             if (auto store = dyn_cast<cudaq::cc::StoreOp>(op))
               return store.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>()
                          ? WalkResult::advance()
                          : WalkResult::interrupt();
             auto name = op->getName().getStringRef();
             if (name.starts_with("arith.") || name.starts_with("math.") ||
                 name == "quake.extract_ref" || name == "quake.veq_size" ||
                 name == "quake.subveq" || name == "quake.relax_size")
               return WalkResult::advance();
             return WalkResult::interrupt();
           })
           .wasInterrupted();
  active.erase(prep);
  return eligible;
}

struct LowerFrequencies
    : cudaq::opt::impl::LowerFrequenciesBase<LowerFrequencies> {
  using LowerFrequenciesBase::LowerFrequenciesBase;
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(&getContext());
    auto i64 = b.getI64Type();
    auto ptr = cudaq::cc::PointerType::get(b.getI8Type());
    auto declare = [&](StringRef name, TypeRange inputs, TypeRange results) {
      if (!module.lookupSymbol(name)) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(module.getBody());
        auto fn = func::FuncOp::create(b, module.getLoc(), name,
                                       b.getFunctionType(inputs, results));
        fn.setPrivate();
      }
    };
    SmallVector<cudaq::quake::FrequenciesOp> requests;
    module.walk(
        [&](cudaq::quake::FrequenciesOp op) { requests.push_back(op); });
    for (auto request : requests)
      request->getParentOfType<func::FuncOp>()->setAttr(
          "cudaq.frequencies.entry", b.getUnitAttr());
    bool hasEmissions = false;
    module.walk([&](cudaq::quake::EmitOp emission) {
      hasEmissions = true;
      emission->getParentOfType<func::FuncOp>()->setAttr(
          "cudaq.frequencies.entry", b.getUnitAttr());
    });
    if ((!requests.empty() || hasEmissions) &&
        !module->hasAttr("cudaq.frequencies.source")) {
      std::string source;
      llvm::raw_string_ostream out(source);
      module.print(out);
      module->setAttr("cudaq.frequencies.source", b.getStringAttr(source));
    }
    // These preparations are device-only functions with borrowed quantum
    // parameters. They must not acquire host launch registrations.
    if (!requests.empty()) {
      constexpr StringLiteral mapName = "quake.mangled_name_map";
      if (auto map = module->getAttrOfType<DictionaryAttr>(mapName)) {
        NamedAttrList names(map);
        for (auto op : requests)
          names.erase(op.getPreparation());
        module->setAttr(mapName, names.getDictionary(&getContext()));
      }
    }
    for (auto op : requests) {
      auto prep = module.lookupSymbol<func::FuncOp>(op.getPreparation());
      auto regType = cast<cudaq::quake::VeqType>(prep.getArgumentTypes()[0]);
      declare("__cudaq_histogram_create", {i64}, {ptr});
      declare("__cudaq_histogram_bit", {ptr, b.getI1Type()}, {});
      declare("__cudaq_histogram_count", {ptr}, {});
      b.setInsertionPoint(op);
      auto loc = op.getLoc();
      Value histogram = func::CallOp::create(b, loc, TypeRange{ptr},
                                             "__cudaq_histogram_create",
                                             ValueRange{op.getShots()})
                            .getResult(0);
      auto decompose = [&](OpBuilder &b) {
        cudaq::opt::factory::createInvariantLoop(
            b, loc, op.getShots(),
            [&](OpBuilder &b, Location loc, Region &, Block &) {
              Value q = cudaq::quake::AllocaOp::create(b, loc, regType);
              SmallVector<Value> args{q};
              llvm::append_range(args, op.getArgs());
              func::CallOp::create(b, loc, prep, args);
              for (std::size_t index = 0; index < regType.getSize(); ++index) {
                Value bit =
                    cudaq::quake::ExtractRefOp::create(b, loc, q, index);
                Value measured =
                    cudaq::quake::MzOp::create(
                        b, loc, cudaq::quake::MeasureType::get(&getContext()),
                        bit)
                        .getResult(0);
                Value value = cudaq::quake::DiscriminateOp::create(
                    b, loc, b.getI1Type(), measured);
                func::CallOp::create(b, loc, TypeRange{},
                                     "__cudaq_histogram_bit",
                                     ValueRange{histogram, value});
              }
              func::CallOp::create(b, loc, TypeRange{},
                                   "__cudaq_histogram_count",
                                   ValueRange{histogram});
              cudaq::quake::DeallocOp::create(b, loc, q);
            });
      };
      llvm::SmallPtrSet<Operation *, 8> active;
      if (isUnitaryPreparation(prep, active)) {
        declare("__nvqir__frequencies_available", {i64}, {b.getI1Type()});
        auto runtimeRegType = cudaq::quake::VeqType::getUnsized(&getContext());
        declare("__nvqir__frequencies_sample", {runtimeRegType, i64}, {ptr});
        declare("__cudaq_histogram_assign", {ptr, ptr}, {});
        b.setInsertionPoint(op);
        Value available = func::CallOp::create(b, loc, TypeRange{b.getI1Type()},
                                               "__nvqir__frequencies_available",
                                               ValueRange{op.getShots()})
                              .getResult(0);
        cudaq::cc::IfOp::create(
            b, loc, TypeRange{}, available,
            [&](OpBuilder &b, Location loc, Region &region) {
              cudaq::cc::RegionBuilderGuard guard(b, loc, region, TypeRange{});
              Value q = cudaq::quake::AllocaOp::create(b, loc, regType);
              SmallVector<Value> args{q};
              llvm::append_range(args, op.getArgs());
              func::CallOp::create(b, loc, prep, args);
              Value readout =
                  cudaq::quake::RelaxSizeOp::create(b, loc, runtimeRegType, q);
              Value result =
                  func::CallOp::create(b, loc, TypeRange{ptr},
                                       "__nvqir__frequencies_sample",
                                       ValueRange{readout, op.getShots()})
                      .getResult(0);
              func::CallOp::create(b, loc, TypeRange{},
                                   "__cudaq_histogram_assign",
                                   ValueRange{histogram, result});
              cudaq::quake::DeallocOp::create(b, loc, q);
              cudaq::cc::ContinueOp::create(b, loc);
            },
            [&](OpBuilder &b, Location loc, Region &region) {
              cudaq::cc::RegionBuilderGuard guard(b, loc, region, TypeRange{});
              decompose(b);
              cudaq::cc::ContinueOp::create(b, loc);
            });
      } else {
        decompose(b);
      }
      op.getHistogram().replaceAllUsesWith(histogram);
      op.erase();
    }
    SmallVector<cudaq::quake::EmitOp> emissions;
    module.walk([&](cudaq::quake::EmitOp op) { emissions.push_back(op); });
    for (auto op : emissions) {
      auto value = op.getValue();
      auto name = isa<cudaq::cc::PointerType>(value.getType())
                      ? "__cudaq_emit_histogram"
                      : "__cudaq_emit_integer";
      if (value.getType() != ptr && value.getType() != i64) {
        op.emitError("prototype emit supports histograms and int64 values");
        signalPassFailure();
        return;
      }
      declare(name, {value.getType()}, {});
      b.setInsertionPoint(op);
      func::CallOp::create(b, op.getLoc(), TypeRange{}, name,
                           ValueRange{value});
      op.erase();
    }
  }
};
} // namespace
