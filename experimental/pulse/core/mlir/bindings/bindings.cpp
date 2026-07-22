// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// nanobind Python bindings for cudaq-pulse MLIR dialects and passes.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <bit>
#include <cstring>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"

namespace nb = nanobind;

// Conversion passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();
}
namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
}
namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
}
// Pulse Transforms passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass();
std::unique_ptr<mlir::Pass> createPulseCanonicalizePass();
std::unique_ptr<mlir::Pass> createVirtualZPass();
std::unique_ptr<mlir::Pass> createPulseFusionPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
} // namespace pulse

namespace {

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry reg;
  reg.insert<pulse::PulseDialect>();
  reg.insert<qop::QOpDialect>();
  reg.insert<cudm::CuDensityMatDialect>();
  reg.insert<mlir::arith::ArithDialect>();
  reg.insert<mlir::func::FuncDialect>();
  reg.insert<mlir::scf::SCFDialect>();
  reg.insert<mlir::LLVM::LLVMDialect>();
  return reg;
}

// Packed-buffer OpCode enum — must match Python packed_emit.py constants
constexpr int kOpAllocDrive = 0;
constexpr int kOpAllocReadout = 1;
constexpr int kOpAllocTone = 2;
constexpr int kOpWfGaussian = 3;
constexpr int kOpWfSquare = 4;
constexpr int kOpWfDrag = 5;
constexpr int kOpWfCosine = 6;
constexpr int kOpWfTanhRamp = 7;
constexpr int kOpWfGaussSquare = 8;
constexpr int kOpWfCustom = 9;
constexpr int kOpDrive = 10;
constexpr int kOpReadout = 11;
constexpr int kOpSync = 12;
constexpr int kOpWait = 13;
constexpr int kOpShiftPhase = 14;
constexpr int kOpSetPhase = 15;
constexpr int kOpShiftFreq = 16;
constexpr int kOpSetFreq = 17;
constexpr int kOpParam = 18;

inline double i2f(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

// -----------------------------------------------------------------------
// PulseModule: opaque handle around an in-memory mlir::ModuleOp
// -----------------------------------------------------------------------
class PulseModule {
public:
  PulseModule(std::shared_ptr<mlir::MLIRContext> ctx,
              mlir::OwningOpRef<mlir::ModuleOp> mod)
      : ctx_(std::move(ctx)), module_(std::move(mod)) {}

  std::string print() {
    std::string result;
    llvm::raw_string_ostream os(result);
    module_->print(os);
    return result;
  }

  std::string run_passes(const std::vector<std::string> &pipeline) {
    mlir::PassManager pm(ctx_.get());
    for (auto &name : pipeline) {
      if (name == "pulse-to-qop")
        pm.addPass(pulse::createPulseToQOpPass());
      else if (name == "qop-to-cudm")
        pm.addPass(qop::createQOpToCuDensityMatPass());
      else if (name == "cudm-to-llvm")
        pm.addPass(cudm::createCuDensityMatToLLVMPass());
      else if (name == "pulse-verify")
        pm.addPass(pulse::createPulseVerifyPass());
      else if (name == "pulse-canonicalize")
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseCanonicalizePass());
      else if (name == "pulse-virtual-z")
        pm.addNestedPass<mlir::func::FuncOp>(pulse::createVirtualZPass());
      else if (name == "pulse-fusion")
        pm.addNestedPass<mlir::func::FuncOp>(pulse::createPulseFusionPass());
      else if (name == "pulse-schedule-alap")
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseScheduleAlapPass());
      else if (name == "loop-invariant-code-motion")
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
      else
        throw std::runtime_error("Unknown pass: " + name);
    }
    if (mlir::failed(pm.run(*module_)))
      throw std::runtime_error("MLIR pass pipeline failed");
    return print();
  }

  std::string run_full_lowering() {
    return run_passes({"pulse-to-qop", "qop-to-cudm", "cudm-to-llvm"});
  }

  bool is_parametric() {
    auto funcOp = getFuncOp();
    return funcOp && funcOp.getNumArguments() > 0;
  }

  std::vector<std::string> param_names() {
    std::vector<std::string> names;
    if (auto attr = module_->getOperation()->getAttrOfType<mlir::ArrayAttr>("pulse.param_names")) {
      for (auto a : attr)
        names.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
    }
    return names;
  }

  PulseModule specialize(const std::vector<double> &f64_args,
                         const std::vector<int64_t> &i64_args) {
    auto clonedModule = module_->clone();
    auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp)
      throw std::runtime_error("specialize: no 'main' func in module");

    auto &entryBlock = funcOp.getBody().front();
    auto loc = funcOp.getLoc();
    mlir::OpBuilder builder(ctx_.get());
    builder.setInsertionPointToStart(&entryBlock);

    size_t f64Idx = 0, i64Idx = 0;
    auto f64Ty = builder.getF64Type();
    auto i64Ty = builder.getIntegerType(64);

    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      auto arg = entryBlock.getArgument(i);
      mlir::Value replacement;
      if (arg.getType().isInteger(64)) {
        if (i64Idx >= i64_args.size())
          throw std::runtime_error("specialize: not enough i64 arguments");
        replacement = builder.create<mlir::arith::ConstantIntOp>(
            loc, i64_args[i64Idx++], 64);
      } else {
        if (f64Idx >= f64_args.size())
          throw std::runtime_error("specialize: not enough f64 arguments");
        replacement = builder.create<mlir::arith::ConstantFloatOp>(
            loc, f64Ty, llvm::APFloat(f64_args[f64Idx++]));
      }
      arg.replaceAllUsesWith(replacement);
    }

    // Remove block arguments (make func take no args)
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    funcOp.setType(builder.getFunctionType({}, {}));

    // Run canonicalize + scheduling on the specialized module
    mlir::PassManager pm(ctx_.get());
    pm.addNestedPass<mlir::func::FuncOp>(
        pulse::createPulseScheduleAlapPass());
    if (mlir::failed(pm.run(clonedModule)))
      throw std::runtime_error("specialize: pass pipeline failed");

    auto owning = mlir::OwningOpRef<mlir::ModuleOp>(clonedModule);
    return PulseModule(ctx_, std::move(owning));
  }

private:
  mlir::func::FuncOp getFuncOp() {
    return module_->lookupSymbol<mlir::func::FuncOp>("main");
  }

  std::shared_ptr<mlir::MLIRContext> ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

// -----------------------------------------------------------------------
// PulseModuleBuilder: construct in-memory Pulse IR from Python dicts
//   Also supports streaming/stateful API for Phase D direct emission.
// -----------------------------------------------------------------------
class PulseModuleBuilder {
public:
  PulseModuleBuilder() {
    auto reg = makeRegistry();
    ctx_ = std::make_shared<mlir::MLIRContext>();
    ctx_->appendDialectRegistry(reg);
    ctx_->loadAllAvailableDialects();
    builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
  }

  // ---- Streaming API (Phase D) ----

  void begin_module(const std::string &name) {
    auto loc = builder_->getUnknownLoc();
    module_ = mlir::ModuleOp::create(loc, mlir::StringRef(name));
    auto funcType = builder_->getFunctionType({}, {});
    funcOp_ = mlir::func::FuncOp::create(loc, "main", funcType);
    module_.push_back(funcOp_);
    auto *entryBlock = funcOp_.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);
  }

  int64_t make_qudit() {
    auto loc = builder_->getUnknownLoc();
    auto qrefTy = pulse::QrefType::get(ctx_.get());
    auto op = builder_->create<pulse::QuditAllocOp>(loc, qrefTy);
    return storeVal(op.getResult());
  }

  std::pair<int64_t, int64_t> get_drive_line(int64_t quditHandle) {
    auto loc = builder_->getUnknownLoc();
    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto op = builder_->create<pulse::GetDriveLineOp>(loc, driveTy, toneTy,
                                                      loadVal(quditHandle));
    return {storeVal(op.getLine()), storeVal(op.getTone())};
  }

  std::pair<int64_t, int64_t> get_readout_line(int64_t quditHandle) {
    auto loc = builder_->getUnknownLoc();
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto op = builder_->create<pulse::GetReadoutLineOp>(loc, readoutTy, toneTy,
                                                        loadVal(quditHandle));
    return {storeVal(op.getLine()), storeVal(op.getTone())};
  }

  int64_t emit_gaussian(int64_t duration, double amplitude, double sigma) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amplitude));
    auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
    auto op = builder_->create<pulse::GaussianPulseOp>(
        loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_square(int64_t duration, double ampReal, double ampImag) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto arC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(ampReal));
    auto aiC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(ampImag));
    auto op = builder_->create<pulse::SquarePulseOp>(
        loc, wfTy, durC.getResult(), arC.getResult(), aiC.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_drag(int64_t duration, double amplitude, double sigma,
                    double beta) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amplitude));
    auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
    auto betaC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(beta));
    auto op = builder_->create<pulse::DRAGPulseOp>(
        loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult(), betaC.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_cosine(int64_t duration, double amplitude) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amplitude));
    auto op = builder_->create<pulse::CosinePulseOp>(
        loc, wfTy, durC.getResult(), ampC.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_tanh_ramp(int64_t duration, double amplitude, double sigma) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amplitude));
    auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
    auto op = builder_->create<pulse::TanhRampOp>(
        loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_gaussian_square(int64_t duration, double amplitude, double sigma,
                               int64_t risefall) {
    auto loc = builder_->getUnknownLoc();
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amplitude));
    auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
    auto rfC = builder_->create<mlir::arith::ConstantIntOp>(loc, risefall, 64);
    auto op = builder_->create<pulse::GaussianSquarePulseOp>(
        loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult(), rfC.getResult());
    return storeVal(op.getResult());
  }

  std::pair<int64_t, int64_t> emit_drive(int64_t lineH, int64_t wfH,
                                         int64_t toneH) {
    auto loc = builder_->getUnknownLoc();
    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto op = builder_->create<pulse::DriveOp>(
        loc, driveTy, toneTy, loadVal(lineH), loadVal(wfH), loadVal(toneH));
    return {storeVal(op.getUpdatedLine()), storeVal(op.getUpdatedTone())};
  }

  std::tuple<int64_t, int64_t, int64_t> emit_readout(int64_t lineH, int64_t wfH,
                                                     int64_t toneH,
                                                     const std::string &mode) {
    auto loc = builder_->getUnknownLoc();
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto measTy = pulse::MeasurementType::get(ctx_.get());
    auto op = builder_->create<pulse::ReadoutOp>(
        loc, readoutTy, toneTy, measTy, loadVal(lineH), loadVal(wfH),
        loadVal(toneH), builder_->getStringAttr(mode));
    return {storeVal(op.getUpdatedLine()), storeVal(op.getUpdatedTone()),
            storeVal(op.getResult())};
  }

  int64_t emit_wait(int64_t lineH, int64_t duration) {
    auto loc = builder_->getUnknownLoc();
    auto lineVal = loadVal(lineH);
    auto durTy = pulse::DurationType::get(ctx_.get());
    auto durConst =
        builder_->create<mlir::arith::ConstantIntOp>(loc, duration, 64);
    auto durOp = builder_->create<pulse::DurationFromIntOp>(
        loc, durTy, durConst.getResult());
    auto waitOp = builder_->create<pulse::WaitOp>(loc, lineVal.getType(),
                                                  lineVal, durOp.getResult());
    return storeVal(waitOp.getResult());
  }

  std::vector<int64_t> emit_sync(const std::vector<int64_t> &lineHandles) {
    auto loc = builder_->getUnknownLoc();
    llvm::SmallVector<mlir::Value> inVals;
    llvm::SmallVector<mlir::Type> outTypes;
    for (auto h : lineHandles) {
      auto v = loadVal(h);
      inVals.push_back(v);
      outTypes.push_back(v.getType());
    }
    auto syncOp = builder_->create<pulse::SyncOp>(loc, outTypes, inVals);
    std::vector<int64_t> results;
    for (unsigned i = 0; i < syncOp.getNumResults(); ++i)
      results.push_back(storeVal(syncOp.getResult(i)));
    return results;
  }

  int64_t emit_shift_phase(int64_t toneH, double delta) {
    auto loc = builder_->getUnknownLoc();
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto phConst = builder_->create<mlir::arith::ConstantFloatOp>(
        loc, f64Ty, llvm::APFloat(delta));
    auto op = builder_->create<pulse::ShiftPhaseOp>(loc, toneTy, loadVal(toneH),
                                                    phConst.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_set_phase(int64_t toneH, double phase) {
    auto loc = builder_->getUnknownLoc();
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto phConst = builder_->create<mlir::arith::ConstantFloatOp>(
        loc, f64Ty, llvm::APFloat(phase));
    auto op = builder_->create<pulse::SetPhaseOp>(loc, toneTy, loadVal(toneH),
                                                  phConst.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_shift_frequency(int64_t toneH, double freqHz) {
    auto loc = builder_->getUnknownLoc();
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto fConst = builder_->create<mlir::arith::ConstantFloatOp>(
        loc, f64Ty, llvm::APFloat(freqHz));
    auto op = builder_->create<pulse::ShiftFrequencyOp>(
        loc, toneTy, loadVal(toneH), fConst.getResult());
    return storeVal(op.getResult());
  }

  int64_t emit_set_frequency(int64_t toneH, double freqHz) {
    auto loc = builder_->getUnknownLoc();
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto f64Ty = builder_->getF64Type();
    auto fConst = builder_->create<mlir::arith::ConstantFloatOp>(
        loc, f64Ty, llvm::APFloat(freqHz));
    auto op = builder_->create<pulse::SetFrequencyOp>(
        loc, toneTy, loadVal(toneH), fConst.getResult());
    return storeVal(op.getResult());
  }

  PulseModule finish_module() {
    auto loc = builder_->getUnknownLoc();
    builder_->create<mlir::func::ReturnOp>(loc);
    auto owningModule = mlir::OwningOpRef<mlir::ModuleOp>(module_);
    return PulseModule(ctx_, std::move(owningModule));
  }

  // ---- Batch API (Phase B) ----

  PulseModule build_from_program(nb::dict prog_dict) {
    auto loc = builder_->getUnknownLoc();

    std::string name = "main";
    if (prog_dict.contains("name"))
      name = nb::cast<std::string>(prog_dict["name"]);

    auto module = mlir::ModuleOp::create(loc, mlir::StringRef(name));

    double clock_ghz = 2.0;
    if (prog_dict.contains("clock_ghz"))
      clock_ghz = nb::cast<double>(prog_dict["clock_ghz"]);

    // Module-level attributes
    if (prog_dict.contains("qubit_freq_hz")) {
      nb::dict freq = nb::cast<nb::dict>(prog_dict["qubit_freq_hz"]);
      int64_t n_qubits = nb::len(freq);
      module->setAttr("pulse.n_qubits", builder_->getI64IntegerAttr(n_qubits));
      module->setAttr("pulse.clock_ghz", builder_->getF64FloatAttr(clock_ghz));
    }

    auto funcType = builder_->getFunctionType({}, {});
    auto funcOp = mlir::func::FuncOp::create(loc, "main", funcType);
    module.push_back(funcOp);

    auto *entryBlock = funcOp.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);

    // Type lookups
    auto qrefTy = pulse::QrefType::get(ctx_.get());
    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto durTy = pulse::DurationType::get(ctx_.get());
    auto measTy = pulse::MeasurementType::get(ctx_.get());

    // SSA value table: vid -> mlir::Value
    llvm::DenseMap<int64_t, mlir::Value> ssaTable;

    // Track qubits
    llvm::DenseMap<int64_t, mlir::Value> qubitSSA;

    auto bindResult = [&](int64_t vid, mlir::Value val) {
      ssaTable[vid] = val;
    };
    auto lookupVal = [&](int64_t vid) -> mlir::Value {
      auto it = ssaTable.find(vid);
      if (it == ssaTable.end())
        throw std::runtime_error("SSA value not found for vid=" +
                                 std::to_string(vid));
      return it->second;
    };

    // Process ops
    nb::list ops = nb::cast<nb::list>(prog_dict["ops"]);
    for (size_t i = 0; i < nb::len(ops); ++i) {
      nb::dict opDict = nb::cast<nb::dict>(ops[i]);
      std::string kind = nb::cast<std::string>(opDict["kind"]);
      nb::dict attrs = nb::cast<nb::dict>(opDict["attrs"]);
      nb::list results = nb::cast<nb::list>(opDict["results"]);
      nb::list operands = nb::cast<nb::list>(opDict["operands"]);

      if (kind == "alloc_drive_line" || kind == "alloc_drive") {
        int64_t qubit = nb::cast<int64_t>(attrs["qubit"]);
        mlir::Value qref;
        auto qit = qubitSSA.find(qubit);
        if (qit == qubitSSA.end()) {
          qref = builder_->create<pulse::QuditAllocOp>(loc, qrefTy).getResult();
          qubitSSA[qubit] = qref;
        } else {
          qref = qit->second;
        }
        auto gdl =
            builder_->create<pulse::GetDriveLineOp>(loc, driveTy, toneTy, qref);
        int64_t lineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[1])["vid"]);
        bindResult(lineVid, gdl.getLine());
        bindResult(toneVid, gdl.getTone());

      } else if (kind == "alloc_readout_line" || kind == "alloc_readout") {
        int64_t qubit = nb::cast<int64_t>(attrs["qubit"]);
        mlir::Value qref;
        auto qit = qubitSSA.find(qubit);
        if (qit == qubitSSA.end()) {
          qref = builder_->create<pulse::QuditAllocOp>(loc, qrefTy).getResult();
          qubitSSA[qubit] = qref;
        } else {
          qref = qit->second;
        }
        auto grl = builder_->create<pulse::GetReadoutLineOp>(loc, readoutTy,
                                                             toneTy, qref);
        int64_t lineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[1])["vid"]);
        bindResult(lineVid, grl.getLine());
        bindResult(toneVid, grl.getTone());

      } else if (kind == "alloc_tone") {
        double freq = 0.0;
        if (attrs.contains("frequency_hz"))
          freq = nb::cast<double>(attrs["frequency_hz"]);
        double phase = 0.0;
        if (attrs.contains("phase_rad"))
          phase = nb::cast<double>(attrs["phase_rad"]);
        auto freqConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto phaseConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(phase));
        auto toneOp = builder_->create<pulse::ToneOp>(
            loc, toneTy, freqConst.getResult(), phaseConst.getResult());
        int64_t resVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        bindResult(resVid, toneOp.getResult());

      } else if (kind == "make_waveform") {
        std::string wfType = nb::cast<std::string>(attrs["waveform_type"]);
        int64_t resVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);

        auto f64Ty = builder_->getF64Type();
        auto i64Ty = builder_->getIntegerType(64);

        if (wfType == "gaussian") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          double amp = extractReal(attrs["amplitude"]);
          double sigma = nb::cast<double>(attrs["sigma"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amp));
          auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
          auto op = builder_->create<pulse::GaussianPulseOp>(
              loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult());
          bindResult(resVid, op.getResult());

        } else if (wfType == "square") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          auto ampPair = extractComplexPair(attrs["amplitude"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto arC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(ampPair.first));
          auto aiC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(ampPair.second));
          auto op = builder_->create<pulse::SquarePulseOp>(
              loc, wfTy, durC.getResult(), arC.getResult(), aiC.getResult());
          bindResult(resVid, op.getResult());

        } else if (wfType == "drag") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          double amp = extractReal(attrs["amplitude"]);
          double sigma = nb::cast<double>(attrs["sigma"]);
          double beta = nb::cast<double>(attrs["beta"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amp));
          auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
          auto betaC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(beta));
          auto op = builder_->create<pulse::DRAGPulseOp>(
              loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult(), betaC.getResult());
          bindResult(resVid, op.getResult());

        } else if (wfType == "cosine") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          double amp = extractReal(attrs["amplitude"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amp));
          auto op = builder_->create<pulse::CosinePulseOp>(
              loc, wfTy, durC.getResult(), ampC.getResult());
          bindResult(resVid, op.getResult());

        } else if (wfType == "tanh_ramp") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          double amp = extractReal(attrs["amplitude"]);
          double sigma = nb::cast<double>(attrs["sigma"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amp));
          auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
          auto op = builder_->create<pulse::TanhRampOp>(
              loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult());
          bindResult(resVid, op.getResult());

        } else if (wfType == "gaussian_square") {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          double amp = extractReal(attrs["amplitude"]);
          double sigma = nb::cast<double>(attrs["sigma"]);
          int64_t risefall = nb::cast<int64_t>(attrs["risefall"]);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto ampC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(amp));
          auto sigC = builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(sigma));
          auto rfC = builder_->create<mlir::arith::ConstantIntOp>(loc, risefall, 64);
          auto op = builder_->create<pulse::GaussianSquarePulseOp>(
              loc, wfTy, durC.getResult(), ampC.getResult(), sigC.getResult(), rfC.getResult());
          bindResult(resVid, op.getResult());

        } else {
          int64_t dur = nb::cast<int64_t>(attrs["duration_vtu"]);
          auto callee = mlir::FlatSymbolRefAttr::get(ctx_.get(), wfType);
          auto durC = builder_->create<mlir::arith::ConstantIntOp>(loc, dur, 64);
          auto op = builder_->create<pulse::CustomOp>(
              loc, wfTy, callee, durC.getResult());
          bindResult(resVid, op.getResult());
        }

      } else if (kind == "drive") {
        int64_t lineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        int64_t wfVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[1])["vid"]);
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[2])["vid"]);
        auto driveOp = builder_->create<pulse::DriveOp>(
            loc, driveTy, toneTy, lookupVal(lineVid), lookupVal(wfVid),
            lookupVal(toneVid));
        // Scheduling attrs
        if (attrs.contains("start_vtu"))
          driveOp->setAttr("start_vtu",
                           builder_->getI64IntegerAttr(
                               nb::cast<int64_t>(attrs["start_vtu"])));
        if (attrs.contains("duration_vtu"))
          driveOp->setAttr("duration_vtu",
                           builder_->getI64IntegerAttr(
                               nb::cast<int64_t>(attrs["duration_vtu"])));
        int64_t rLineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        int64_t rToneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[1])["vid"]);
        bindResult(rLineVid, driveOp.getUpdatedLine());
        bindResult(rToneVid, driveOp.getUpdatedTone());

      } else if (kind == "readout") {
        int64_t lineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        int64_t wfVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[1])["vid"]);
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[2])["vid"]);
        std::string mode = "iq";
        if (attrs.contains("mode"))
          mode = nb::cast<std::string>(attrs["mode"]);
        auto roOp = builder_->create<pulse::ReadoutOp>(
            loc, readoutTy, toneTy, measTy, lookupVal(lineVid),
            lookupVal(wfVid), lookupVal(toneVid),
            builder_->getStringAttr(mode));
        int64_t rLineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        int64_t rToneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[1])["vid"]);
        int64_t rMeasVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[2])["vid"]);
        bindResult(rLineVid, roOp.getUpdatedLine());
        bindResult(rToneVid, roOp.getUpdatedTone());
        bindResult(rMeasVid, roOp.getResult());

      } else if (kind == "wait") {
        int64_t lineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        int64_t durVtu = 0;
        if (attrs.contains("duration_vtu"))
          durVtu = nb::cast<int64_t>(attrs["duration_vtu"]);
        mlir::Value lineVal = lookupVal(lineVid);
        auto lineType = lineVal.getType();
        auto durConst =
            builder_->create<mlir::arith::ConstantIntOp>(loc, durVtu, 64);
        auto durOp = builder_->create<pulse::DurationFromIntOp>(
            loc, durTy, durConst.getResult());
        auto waitOp = builder_->create<pulse::WaitOp>(loc, lineType, lineVal,
                                                      durOp.getResult());
        int64_t rLineVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
        bindResult(rLineVid, waitOp.getResult());

      } else if (kind == "sync") {
        llvm::SmallVector<mlir::Value> inVals;
        llvm::SmallVector<mlir::Type> outTypes;
        for (size_t j = 0; j < nb::len(operands); ++j) {
          int64_t vid =
              nb::cast<int64_t>(nb::cast<nb::dict>(operands[j])["vid"]);
          inVals.push_back(lookupVal(vid));
        }
        for (size_t j = 0; j < nb::len(results); ++j) {
          nb::dict rd = nb::cast<nb::dict>(results[j]);
          std::string vtype = nb::cast<std::string>(rd["vtype"]);
          if (vtype == "drive_line")
            outTypes.push_back(driveTy);
          else if (vtype == "readout_line")
            outTypes.push_back(readoutTy);
          else
            outTypes.push_back(driveTy);
        }
        auto syncOp = builder_->create<pulse::SyncOp>(loc, outTypes, inVals);
        for (size_t j = 0; j < nb::len(results); ++j) {
          int64_t vid =
              nb::cast<int64_t>(nb::cast<nb::dict>(results[j])["vid"]);
          bindResult(vid, syncOp.getResult(j));
        }

      } else if (kind == "shift_phase") {
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        double delta = 0.0;
        if (attrs.contains("delta_rad"))
          delta = nb::cast<double>(attrs["delta_rad"]);
        else if (attrs.contains("delta"))
          delta = nb::cast<double>(attrs["delta"]);
        auto phConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(delta));
        auto spOp = builder_->create<pulse::ShiftPhaseOp>(
            loc, toneTy, lookupVal(toneVid), phConst.getResult());
        if (nb::len(results) > 0) {
          int64_t rVid =
              nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
          bindResult(rVid, spOp.getResult());
        }

      } else if (kind == "set_phase") {
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        double phase = 0.0;
        if (attrs.contains("phase_rad"))
          phase = nb::cast<double>(attrs["phase_rad"]);
        auto phConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(phase));
        auto spOp = builder_->create<pulse::SetPhaseOp>(
            loc, toneTy, lookupVal(toneVid), phConst.getResult());
        if (nb::len(results) > 0) {
          int64_t rVid =
              nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
          bindResult(rVid, spOp.getResult());
        }

      } else if (kind == "shift_frequency") {
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        double freq = 0.0;
        if (attrs.contains("delta_hz"))
          freq = nb::cast<double>(attrs["delta_hz"]);
        else if (attrs.contains("freq_hz"))
          freq = nb::cast<double>(attrs["freq_hz"]);
        auto fConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto sfOp = builder_->create<pulse::ShiftFrequencyOp>(
            loc, toneTy, lookupVal(toneVid), fConst.getResult());
        if (nb::len(results) > 0) {
          int64_t rVid =
              nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
          bindResult(rVid, sfOp.getResult());
        }

      } else if (kind == "set_frequency") {
        int64_t toneVid =
            nb::cast<int64_t>(nb::cast<nb::dict>(operands[0])["vid"]);
        double freq = 0.0;
        if (attrs.contains("freq_hz"))
          freq = nb::cast<double>(attrs["freq_hz"]);
        auto fConst = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto sfOp = builder_->create<pulse::SetFrequencyOp>(
            loc, toneTy, lookupVal(toneVid), fConst.getResult());
        if (nb::len(results) > 0) {
          int64_t rVid =
              nb::cast<int64_t>(nb::cast<nb::dict>(results[0])["vid"]);
          bindResult(rVid, sfOp.getResult());
        }

      } else if (kind == "for_loop" || kind == "end_for") {
        // Loops are handled by the scheduling pass and by Phase D's
        // streaming builder. The batch builder skips them.
      }
    }

    // Return terminator
    builder_->create<mlir::func::ReturnOp>(loc);

    auto owningModule = mlir::OwningOpRef<mlir::ModuleOp>(module);
    return PulseModule(ctx_, std::move(owningModule));
  }

  // ─── Packed-buffer decoder (zero-copy path) ──────────────────────────
  PulseModule build_from_packed(
      nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> stream, double clock_ghz,
      int64_t n_qubits,
      nb::ndarray<double, nb::ndim<1>, nb::c_contig> qubit_freqs,
      std::vector<std::string> param_names = {},
      std::vector<std::string> param_types = {}) {

    const int64_t *d = stream.data();
    const size_t len = stream.shape(0);
    auto loc = builder_->getUnknownLoc();

    auto module = mlir::ModuleOp::create(loc, mlir::StringRef("main"));
    module->setAttr("pulse.n_qubits", builder_->getI64IntegerAttr(n_qubits));
    module->setAttr("pulse.clock_ghz", builder_->getF64FloatAttr(clock_ghz));

    // Encode qubit frequencies as module attribute
    const double *fdata = qubit_freqs.data();
    size_t flen = qubit_freqs.shape(0);
    llvm::SmallVector<double> freqVec(fdata, fdata + flen);
    module->setAttr("pulse.qubit_freq_hz",
                    builder_->getF64ArrayAttr(llvm::ArrayRef<double>(freqVec)));

    // Build func.func type: parameters become block arguments
    llvm::SmallVector<mlir::Type> argTypes;
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    for (auto &pt : param_types) {
      if (pt == "i64")
        argTypes.push_back(i64Ty);
      else
        argTypes.push_back(f64Ty);
    }

    auto funcType = builder_->getFunctionType(argTypes, {});
    auto funcOp = mlir::func::FuncOp::create(loc, "main", funcType);
    module.push_back(funcOp);
    auto *entryBlock = funcOp.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);

    // Store parameter name metadata on the module for later use
    if (!param_names.empty()) {
      llvm::SmallVector<mlir::Attribute> nameAttrs;
      for (auto &n : param_names)
        nameAttrs.push_back(builder_->getStringAttr(n));
      module->setAttr("pulse.param_names",
                      builder_->getArrayAttr(nameAttrs));
    }

    auto qrefTy = pulse::QrefType::get(ctx_.get());
    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto durTy = pulse::DurationType::get(ctx_.get());
    auto measTy = pulse::MeasurementType::get(ctx_.get());

    llvm::DenseMap<int64_t, mlir::Value> ssa;
    llvm::DenseMap<int64_t, mlir::Value> qubitSSA;

    auto bind = [&](int64_t vid, mlir::Value v) { ssa[vid] = v; };
    auto look = [&](int64_t vid) -> mlir::Value {
      auto it = ssa.find(vid);
      if (it == ssa.end())
        throw std::runtime_error("packed: SSA value not found for vid=" +
                                 std::to_string(vid));
      return it->second;
    };
    auto getQubit = [&](int64_t q) -> mlir::Value {
      auto it = qubitSSA.find(q);
      if (it != qubitSSA.end())
        return it->second;
      auto v = builder_->create<pulse::QuditAllocOp>(loc, qrefTy).getResult();
      qubitSSA[q] = v;
      return v;
    };

    size_t cur = 0;
    while (cur < len) {
      int64_t hdr = d[cur];
      int opcode = static_cast<int>(hdr & 0xFF);
      int plen = static_cast<int>((hdr >> 8) & 0xFF);
      int pmask = static_cast<int>((hdr >> 16) & 0xFFFF);
      const int64_t *p = d + cur + 1;

      // Helper: get SSA value for a waveform arg slot. If the param_mask
      // bit is set, the slot holds a vid (block arg ref); otherwise it's
      // a literal that needs an arith.constant.
      auto getI64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return builder_->create<mlir::arith::ConstantIntOp>(loc, p[slot], 64)
            .getResult();
      };
      auto getF64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return builder_->create<mlir::arith::ConstantFloatOp>(
                   loc, f64Ty, llvm::APFloat(i2f(p[slot])))
            .getResult();
      };

      switch (opcode) {

      case kOpAllocDrive: {
        auto qref = getQubit(p[0]);
        auto gdl =
            builder_->create<pulse::GetDriveLineOp>(loc, driveTy, toneTy, qref);
        bind(p[1], gdl.getLine());
        bind(p[2], gdl.getTone());
        break;
      }

      case kOpAllocReadout: {
        auto qref = getQubit(p[0]);
        auto grl = builder_->create<pulse::GetReadoutLineOp>(loc, readoutTy,
                                                             toneTy, qref);
        bind(p[1], grl.getLine());
        bind(p[2], grl.getTone());
        break;
      }

      case kOpAllocTone: {
        double freq = i2f(p[1]);
        double phase = i2f(p[2]);
        auto fc = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto pc = builder_->create<mlir::arith::ConstantFloatOp>(
            loc, builder_->getF64Type(), llvm::APFloat(phase));
        auto t = builder_->create<pulse::ToneOp>(loc, toneTy, fc.getResult(),
                                                 pc.getResult());
        bind(p[0], t.getResult());
        break;
      }

      case kOpWfGaussian: {
        auto op = builder_->create<pulse::GaussianPulseOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfSquare: {
        auto op = builder_->create<pulse::SquarePulseOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfDrag: {
        auto op = builder_->create<pulse::DRAGPulseOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3), getF64Arg(4));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCosine: {
        auto op = builder_->create<pulse::CosinePulseOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfTanhRamp: {
        auto op = builder_->create<pulse::TanhRampOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfGaussSquare: {
        auto op = builder_->create<pulse::GaussianSquarePulseOp>(
            loc, wfTy, getI64Arg(1), getF64Arg(2), getF64Arg(3), getI64Arg(4));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCustom: {
        auto callee = mlir::FlatSymbolRefAttr::get(ctx_.get(), "custom");
        auto op = builder_->create<pulse::CustomOp>(
            loc, wfTy, callee, getI64Arg(1));
        bind(p[0], op.getResult());
        break;
      }

      case kOpParam: {
        // Parameter reference: p[0] = vid, p[1] = param_index
        // The value comes from the func.func block argument
        int64_t paramIdx = p[1];
        bind(p[0], entryBlock->getArgument(paramIdx));
        break;
      }

      case kOpDrive: {
        auto driveOp = builder_->create<pulse::DriveOp>(
            loc, driveTy, toneTy, look(p[0]), look(p[1]), look(p[2]));
        if (p[5] != -1)
          driveOp->setAttr("start_vtu", builder_->getI64IntegerAttr(p[5]));
        if (p[6] != -1)
          driveOp->setAttr("duration_vtu", builder_->getI64IntegerAttr(p[6]));
        bind(p[3], driveOp.getUpdatedLine());
        bind(p[4], driveOp.getUpdatedTone());
        break;
      }

      case kOpReadout: {
        auto mode = builder_->getStringAttr("iq");
        auto roOp = builder_->create<pulse::ReadoutOp>(
            loc, readoutTy, toneTy, measTy, look(p[0]), look(p[1]), look(p[2]),
            mode);
        bind(p[3], roOp.getUpdatedLine());
        bind(p[4], roOp.getUpdatedTone());
        bind(p[5], roOp.getResult());
        break;
      }

      case kOpSync: {
        int64_t n = p[0];
        llvm::SmallVector<mlir::Value> inVals;
        llvm::SmallVector<mlir::Type> outTypes;
        for (int64_t j = 0; j < n; ++j) {
          inVals.push_back(look(p[1 + 3 * j]));
          int64_t vtype = p[3 + 3 * j];
          outTypes.push_back(vtype == 1 ? mlir::Type(readoutTy)
                                        : mlir::Type(driveTy));
        }
        auto syncOp = builder_->create<pulse::SyncOp>(loc, outTypes, inVals);
        for (int64_t j = 0; j < n; ++j)
          bind(p[2 + 3 * j], syncOp.getResult(j));
        break;
      }

      case kOpWait: {
        mlir::Value lineVal = look(p[0]);
        mlir::Value durCycles;
        if (pmask & (1 << 2))
          durCycles = look(p[2]);
        else
          durCycles = builder_->create<mlir::arith::ConstantIntOp>(loc, p[2], 64).getResult();
        auto durOp = builder_->create<pulse::DurationFromIntOp>(
            loc, durTy, durCycles);
        auto waitOp = builder_->create<pulse::WaitOp>(
            loc, lineVal.getType(), lineVal, durOp.getResult());
        bind(p[1], waitOp.getResult());
        break;
      }

      case kOpShiftPhase: {
        mlir::Value phaseVal = (pmask & (1 << 2))
            ? look(p[2])
            : builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(i2f(p[2]))).getResult();
        auto spOp = builder_->create<pulse::ShiftPhaseOp>(
            loc, toneTy, look(p[0]), phaseVal);
        bind(p[1], spOp.getResult());
        break;
      }

      case kOpSetPhase: {
        mlir::Value phaseVal = (pmask & (1 << 2))
            ? look(p[2])
            : builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(i2f(p[2]))).getResult();
        auto spOp = builder_->create<pulse::SetPhaseOp>(loc, toneTy, look(p[0]),
                                                        phaseVal);
        bind(p[1], spOp.getResult());
        break;
      }

      case kOpShiftFreq: {
        mlir::Value fVal = (pmask & (1 << 2))
            ? look(p[2])
            : builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(i2f(p[2]))).getResult();
        auto sfOp = builder_->create<pulse::ShiftFrequencyOp>(
            loc, toneTy, look(p[0]), fVal);
        bind(p[1], sfOp.getResult());
        break;
      }

      case kOpSetFreq: {
        mlir::Value fVal = (pmask & (1 << 2))
            ? look(p[2])
            : builder_->create<mlir::arith::ConstantFloatOp>(loc, f64Ty, llvm::APFloat(i2f(p[2]))).getResult();
        auto sfOp = builder_->create<pulse::SetFrequencyOp>(
            loc, toneTy, look(p[0]), fVal);
        bind(p[1], sfOp.getResult());
        break;
      }

      default:
        break;
      }

      cur += 1 + plen;
    }

    builder_->create<mlir::func::ReturnOp>(loc);
    auto owningModule = mlir::OwningOpRef<mlir::ModuleOp>(module);
    return PulseModule(ctx_, std::move(owningModule));
  }

private:
  double extractReal(nb::handle val) {
    if (nb::isinstance<nb::float_>(val))
      return nb::cast<double>(val);
    if (nb::isinstance<nb::int_>(val))
      return static_cast<double>(nb::cast<int64_t>(val));
    // Complex: extract real part
    nb::object obj = nb::borrow(val);
    if (nb::hasattr(obj, "real"))
      return nb::cast<double>(obj.attr("real"));
    return nb::cast<double>(val);
  }

  std::pair<double, double> extractComplexPair(nb::handle val) {
    if (nb::isinstance<nb::float_>(val))
      return {nb::cast<double>(val), 0.0};
    if (nb::isinstance<nb::int_>(val))
      return {static_cast<double>(nb::cast<int64_t>(val)), 0.0};
    nb::object obj = nb::borrow(val);
    if (nb::hasattr(obj, "real") && nb::hasattr(obj, "imag"))
      return {nb::cast<double>(obj.attr("real")),
              nb::cast<double>(obj.attr("imag"))};
    if (nb::isinstance<nb::list>(val) || nb::isinstance<nb::tuple>(val)) {
      nb::list lst = nb::cast<nb::list>(val);
      double re = nb::cast<double>(lst[0]);
      double im = nb::len(lst) > 1 ? nb::cast<double>(lst[1]) : 0.0;
      return {re, im};
    }
    return {nb::cast<double>(val), 0.0};
  }

  int64_t storeVal(mlir::Value val) {
    int64_t handle = nextHandle_++;
    handleTable_[handle] = val;
    return handle;
  }

  mlir::Value loadVal(int64_t handle) {
    auto it = handleTable_.find(handle);
    if (it == handleTable_.end())
      throw std::runtime_error("Invalid MLIR value handle: " +
                               std::to_string(handle));
    return it->second;
  }

  std::shared_ptr<mlir::MLIRContext> ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
  mlir::ModuleOp module_;
  mlir::func::FuncOp funcOp_;
  llvm::DenseMap<int64_t, mlir::Value> handleTable_;
  int64_t nextHandle_ = 0;
};

// -----------------------------------------------------------------------
// MLIRPipeline: text-in / text-out convenience wrapper (legacy API)
// -----------------------------------------------------------------------
class MLIRPipeline {
public:
  MLIRPipeline() : registry_(makeRegistry()) {}

  std::string parse_and_print(const std::string &mlir_text) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &ctx);
    if (!module)
      throw std::runtime_error("Failed to parse MLIR text");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_full_pipeline(const std::string &pulse_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(pulse_mlir, &ctx);
    if (!module)
      throw std::runtime_error("Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(pulse::createPulseToQOpPass());
    pm.addPass(qop::createQOpToCuDensityMatPass());
    pm.addPass(cudm::createCuDensityMatToLLVMPass());
    if (mlir::failed(pm.run(*module)))
      throw std::runtime_error("MLIR pass pipeline failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_pulse_to_qop(const std::string &pulse_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(pulse_mlir, &ctx);
    if (!module)
      throw std::runtime_error("Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(pulse::createPulseToQOpPass());
    if (mlir::failed(pm.run(*module)))
      throw std::runtime_error("pulse-to-qop pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_qop_to_cudm(const std::string &qop_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(qop_mlir, &ctx);
    if (!module)
      throw std::runtime_error("Failed to parse qop MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(qop::createQOpToCuDensityMatPass());
    if (mlir::failed(pm.run(*module)))
      throw std::runtime_error("qop-to-cudm pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_cudm_to_llvm(const std::string &cudm_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(cudm_mlir, &ctx);
    if (!module)
      throw std::runtime_error("Failed to parse cudm MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(cudm::createCuDensityMatToLLVMPass());
    if (mlir::failed(pm.run(*module)))
      throw std::runtime_error("cudm-to-llvm pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  PulseModule parse_to_module(const std::string &mlir_text) {
    auto ctx = std::make_shared<mlir::MLIRContext>();
    ctx->appendDialectRegistry(registry_);
    ctx->loadAllAvailableDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, ctx.get());
    if (!module)
      throw std::runtime_error("Failed to parse MLIR text");
    return PulseModule(ctx, std::move(module));
  }

private:
  mlir::DialectRegistry registry_;
};

} // namespace

NB_MODULE(_cudaq_pulse_native, m) {
  m.doc() = "cudaq-pulse native MLIR pipeline bindings";

  nb::class_<PulseModule>(m, "PulseModule")
      .def("print", &PulseModule::print, "Print MLIR text representation")
      .def("run_passes", &PulseModule::run_passes,
           "Run named MLIR passes on this module")
      .def("run_full_lowering", &PulseModule::run_full_lowering,
           "Run full lowering: pulse -> qop -> cudm -> llvm")
      .def("is_parametric", &PulseModule::is_parametric,
           "True if the module has func.func block arguments (parameters)")
      .def("param_names", &PulseModule::param_names,
           "Get parameter names")
      .def("specialize", &PulseModule::specialize,
           "Clone module, substitute parameters with constants, fold, schedule");

  nb::class_<PulseModuleBuilder>(m, "PulseModuleBuilder")
      .def(nb::init<>())
      .def("build_from_program", &PulseModuleBuilder::build_from_program,
           "Build an in-memory PulseModule from a program dict")
      .def("build_from_packed", &PulseModuleBuilder::build_from_packed,
           "Build an in-memory PulseModule from a packed int64 numpy buffer",
           nb::arg("stream"), nb::arg("clock_ghz"), nb::arg("n_qubits"),
           nb::arg("qubit_freqs"), nb::arg("param_names") = std::vector<std::string>{},
           nb::arg("param_types") = std::vector<std::string>{})
      .def("begin_module", &PulseModuleBuilder::begin_module,
           "Start building a new module")
      .def("make_qudit", &PulseModuleBuilder::make_qudit,
           "Allocate a qudit, returns handle")
      .def("get_drive_line", &PulseModuleBuilder::get_drive_line,
           "Get drive line+tone for a qudit, returns (line_h, tone_h)")
      .def("get_readout_line", &PulseModuleBuilder::get_readout_line,
           "Get readout line+tone for a qudit, returns (line_h, tone_h)")
      .def("emit_gaussian", &PulseModuleBuilder::emit_gaussian,
           "Emit gaussian waveform, returns wf handle")
      .def("emit_square", &PulseModuleBuilder::emit_square,
           "Emit square waveform, returns wf handle")
      .def("emit_drag", &PulseModuleBuilder::emit_drag,
           "Emit DRAG waveform, returns wf handle")
      .def("emit_cosine", &PulseModuleBuilder::emit_cosine,
           "Emit cosine waveform, returns wf handle")
      .def("emit_tanh_ramp", &PulseModuleBuilder::emit_tanh_ramp,
           "Emit tanh_ramp waveform, returns wf handle")
      .def("emit_gaussian_square", &PulseModuleBuilder::emit_gaussian_square,
           "Emit gaussian_square waveform, returns wf handle")
      .def("emit_drive", &PulseModuleBuilder::emit_drive,
           "Emit drive op, returns (line_h, tone_h)")
      .def("emit_readout", &PulseModuleBuilder::emit_readout,
           "Emit readout op, returns (line_h, tone_h, meas_h)")
      .def("emit_wait", &PulseModuleBuilder::emit_wait,
           "Emit wait op, returns line handle")
      .def("emit_sync", &PulseModuleBuilder::emit_sync,
           "Emit sync op, returns list of line handles")
      .def("emit_shift_phase", &PulseModuleBuilder::emit_shift_phase,
           "Emit shift_phase op, returns tone handle")
      .def("emit_set_phase", &PulseModuleBuilder::emit_set_phase,
           "Emit set_phase op, returns tone handle")
      .def("emit_shift_frequency", &PulseModuleBuilder::emit_shift_frequency,
           "Emit shift_frequency op, returns tone handle")
      .def("emit_set_frequency", &PulseModuleBuilder::emit_set_frequency,
           "Emit set_frequency op, returns tone handle")
      .def("finish_module", &PulseModuleBuilder::finish_module,
           "Finalize and return the PulseModule");

  nb::class_<MLIRPipeline>(m, "MLIRPipeline")
      .def(nb::init<>())
      .def("parse_and_print", &MLIRPipeline::parse_and_print,
           "Parse MLIR text and print it back (roundtrip)")
      .def("run_full_pipeline", &MLIRPipeline::run_full_pipeline,
           "Run full pipeline: pulse -> qop -> cudm -> llvm")
      .def("run_pulse_to_qop", &MLIRPipeline::run_pulse_to_qop,
           "Run pulse-to-qop pass only")
      .def("run_qop_to_cudm", &MLIRPipeline::run_qop_to_cudm,
           "Run qop-to-cudm pass only")
      .def("run_cudm_to_llvm", &MLIRPipeline::run_cudm_to_llvm,
           "Run cudm-to-llvm pass only")
      .def("parse_to_module", &MLIRPipeline::parse_to_module,
           "Parse MLIR text into a PulseModule");
}
