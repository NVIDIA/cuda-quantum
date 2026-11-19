/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// PulseToQOp conversion pass: lower pulse dialect ops to qop dialect ops
// for Hamiltonian/Lindblad construction.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include <algorithm>
#include <cmath>
#include <optional>

using namespace mlir;

namespace {

static std::optional<int64_t> traceQubit(Value line,
                                         llvm::DenseSet<Value> &visited) {
  if (!line || !visited.insert(line).second)
    return std::nullopt;
  auto *def = line.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto qubit = def->getAttrOfType<IntegerAttr>("qubit"))
    return qubit.getInt();

  // Sync preserves line identity positionally. Other line-transforming pulse
  // ops carry their physical line in the first operand.
  if (def->getName().getStringRef() == "pulse.sync") {
    auto result = dyn_cast<OpResult>(line);
    if (!result || result.getResultNumber() >= def->getNumOperands())
      return std::nullopt;
    return traceQubit(def->getOperand(result.getResultNumber()), visited);
  }
  for (Value operand : def->getOperands()) {
    if (isa<pulse::DriveLineType, pulse::ReadoutLineType>(operand.getType()))
      return traceQubit(operand, visited);
  }
  return std::nullopt;
}

static std::optional<int64_t> traceQubit(Value line) {
  llvm::DenseSet<Value> visited;
  return traceQubit(line, visited);
}

static std::optional<double> constantNumber(Value value);

struct ToneState {
  double frequencyHz;
  double frameFrequencyHz;
  double phase;
  double timeNs;
};

static std::optional<ToneState> traceTone(Value tone, double clockGHz,
                                          llvm::DenseSet<Value> &visited) {
  if (!tone || !visited.insert(tone).second)
    return std::nullopt;
  Operation *def = tone.getDefiningOp();
  if (!def)
    return std::nullopt;
  StringRef name = def->getName().getStringRef();
  if (name == "pulse.get_drive_line" || name == "pulse.get_readout_line") {
    auto frequency = def->getAttrOfType<FloatAttr>("frequency_hz");
    if (!frequency)
      return std::nullopt;
    double value = frequency.getValueAsDouble();
    return ToneState{value, value, 0.0, 0.0};
  }
  if (name == "pulse.tone" && def->getNumOperands() == 2) {
    auto frequency = constantNumber(def->getOperand(0));
    auto phase = constantNumber(def->getOperand(1));
    if (!frequency || !phase)
      return std::nullopt;
    return ToneState{*frequency, *frequency, *phase, 0.0};
  }
  if (name == "pulse.drive" && def->getNumOperands() >= 3) {
    auto state = traceTone(def->getOperand(2), clockGHz, visited);
    auto start = def->getAttrOfType<IntegerAttr>("start_vtu");
    auto duration = def->getAttrOfType<IntegerAttr>("duration_vtu");
    if (!state || !duration)
      return std::nullopt;
    if (auto framePhase = def->getAttrOfType<FloatAttr>("frame_phase_offset"))
      state->phase += framePhase.getValueAsDouble();
    double endNs =
        static_cast<double>((start ? start.getInt() : 0) + duration.getInt()) /
        clockGHz;
    state->phase += 2.0 * M_PI *
                    (state->frequencyHz - state->frameFrequencyHz) * 1.0e-9 *
                    (endNs - state->timeNs);
    state->timeNs = endNs;
    return state;
  }
  if ((name == "pulse.shift_phase" || name == "pulse.set_phase" ||
       name == "pulse.shift_frequency" || name == "pulse.set_frequency") &&
      def->getNumOperands() == 2) {
    auto state = traceTone(def->getOperand(0), clockGHz, visited);
    auto value = constantNumber(def->getOperand(1));
    if (!state || !value)
      return std::nullopt;
    if (name == "pulse.shift_phase")
      state->phase += *value;
    else if (name == "pulse.set_phase")
      state->phase = *value;
    else if (name == "pulse.shift_frequency")
      state->frequencyHz += *value;
    else
      state->frequencyHz = *value;
    return state;
  }
  return std::nullopt;
}

static std::optional<ToneState> traceTone(Value tone, double clockGHz) {
  llvm::DenseSet<Value> visited;
  return traceTone(tone, clockGHz, visited);
}

static std::optional<double> constantNumber(Value value) {
  Attribute attribute;
  if (!matchPattern(value, m_Constant(&attribute)))
    return std::nullopt;
  if (auto number = dyn_cast<FloatAttr>(attribute))
    return number.getValueAsDouble();
  if (auto number = dyn_cast<IntegerAttr>(attribute))
    return static_cast<double>(number.getInt());
  return std::nullopt;
}

struct WaveformCallbackData {
  int32_t kind;
  SmallVector<double> parameters;
};

static std::optional<WaveformCallbackData>
getWaveformCallbackData(Operation *drive, double clockGHz,
                        double qubitFrequencyHz, double driveScale,
                        int32_t quadrature) {
  if (drive->getNumOperands() < 3)
    return std::nullopt;
  auto *waveform = drive->getOperand(1).getDefiningOp();
  if (!waveform)
    return std::nullopt;
  auto start = drive->getAttrOfType<IntegerAttr>("start_vtu");
  auto duration = drive->getAttrOfType<IntegerAttr>("duration_vtu");
  if (!start || !duration)
    return std::nullopt;

  const double scale = 1.0 / clockGHz;
  const double startNs = static_cast<double>(start.getInt()) * scale;
  const double durationNs = static_cast<double>(duration.getInt()) * scale;
  auto tone = traceTone(drive->getOperand(2), clockGHz);
  if (!tone)
    return std::nullopt;
  const double phaseOffset =
      drive->getAttrOfType<FloatAttr>("phase_offset")
          ? drive->getAttrOfType<FloatAttr>("phase_offset").getValueAsDouble()
          : 0.0;
  const double framePhaseOffset =
      drive->getAttrOfType<FloatAttr>("frame_phase_offset")
          ? drive->getAttrOfType<FloatAttr>("frame_phase_offset")
                .getValueAsDouble()
          : 0.0;
  const double phase =
      tone->phase +
      2.0 * M_PI * (tone->frequencyHz - tone->frameFrequencyHz) * 1.0e-9 *
          (startNs - tone->timeNs) +
      2.0 * M_PI * (tone->frameFrequencyHz - qubitFrequencyHz) * 1.0e-9 *
          startNs +
      phaseOffset + framePhaseOffset;
  const double detuning =
      2.0 * M_PI * (tone->frequencyHz - qubitFrequencyHz) * 1.0e-9;
  // Common layout: start, duration, real amplitude, imaginary amplitude,
  // sigma, beta, rise/fall, phase-at-start, quadrature, detuning (rad/ns),
  // followed by sample data.
  WaveformCallbackData result{0,
                              {startNs, durationNs, 0.0, 0.0, 0.0, 0.0, 0.0,
                               phase, static_cast<double>(quadrature),
                               detuning}};
  auto set = [&](int32_t kind, Value amplitude, Value sigma = {},
                 Value beta = {}) -> bool {
    auto amplitudeValue = constantNumber(amplitude);
    if (!amplitudeValue)
      return false;
    result.kind = kind;
    result.parameters[2] = *amplitudeValue * driveScale;
    if (sigma) {
      auto sigmaValue = constantNumber(sigma);
      if (!sigmaValue)
        return false;
      result.parameters[4] = *sigmaValue * scale;
    }
    if (beta) {
      auto betaValue = constantNumber(beta);
      if (!betaValue)
        return false;
      result.parameters[5] = *betaValue;
    }
    return true;
  };

  if (auto op = dyn_cast<pulse::SquarePulseOp>(waveform)) {
    auto real = constantNumber(op.getAmpReal());
    auto imag = constantNumber(op.getAmpImag());
    if (!real || !imag)
      return std::nullopt;
    result.kind = 1;
    result.parameters[2] = *real * driveScale;
    result.parameters[3] = *imag * driveScale;
  } else if (auto op = dyn_cast<pulse::GaussianPulseOp>(waveform)) {
    if (!set(2, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::DRAGPulseOp>(waveform)) {
    if (!set(3, op.getAmplitude(), op.getSigma(), op.getBeta()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::CosinePulseOp>(waveform)) {
    if (!set(4, op.getAmplitude()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::TanhRampOp>(waveform)) {
    if (!set(5, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
  } else if (auto op = dyn_cast<pulse::GaussianSquarePulseOp>(waveform)) {
    if (!set(6, op.getAmplitude(), op.getSigma()))
      return std::nullopt;
    auto riseFall = constantNumber(op.getRisefall());
    if (!riseFall)
      return std::nullopt;
    result.parameters[6] = *riseFall * scale;
  } else if (auto op = dyn_cast<pulse::CustomSamplesOp>(waveform)) {
    result.kind = 7;
    for (Attribute sample : op.getSamples()) {
      auto value = dyn_cast<FloatAttr>(sample);
      if (!value)
        return std::nullopt;
      result.parameters.push_back(value.getValueAsDouble() * driveScale);
    }
  } else {
    return std::nullopt;
  }
  return result;
}

// ---- Build a qop.spin + qop.const_scalar + qop.make_product term ----
static Value buildStaticTerm(OpBuilder &b, Location loc, Value target,
                             StringRef spinKind, double coeffReal,
                             double coeffImag) {
  auto handlerTy = qop::HandlerType::get(b.getContext());
  auto scalarTy = qop::ScalarType::get(b.getContext());
  auto productTy = qop::ProductType::get(b.getContext());

  // qop.spin
  auto kindAttr = qop::symbolizeHandlerKind(spinKind);
  if (!kindAttr)
    return {};
  auto spin =
      qop::SpinOp::create(b, loc, handlerTy, target,
                          qop::HandlerKindAttr::get(b.getContext(), *kindAttr));

  // qop.const_scalar
  auto scalar =
      qop::ConstScalarOp::create(b, loc, scalarTy, b.getF64FloatAttr(coeffReal),
                                 b.getF64FloatAttr(coeffImag));

  // qop.make_product
  auto product =
      qop::MakeProductOp::create(b, loc, productTy, scalar, ValueRange{spin});

  return product;
}

static Value
buildStaticProduct(OpBuilder &b, Location loc,
                   ArrayRef<std::pair<Value, StringRef>> targetAndKinds,
                   double coefficient) {
  auto handlerTy = qop::HandlerType::get(b.getContext());
  auto scalarTy = qop::ScalarType::get(b.getContext());
  auto productTy = qop::ProductType::get(b.getContext());
  SmallVector<Value> factors;
  for (auto [target, kindName] : targetAndKinds) {
    auto kind = qop::symbolizeHandlerKind(kindName);
    if (!kind)
      return {};
    factors.push_back(
        qop::SpinOp::create(b, loc, handlerTy, target,
                            qop::HandlerKindAttr::get(b.getContext(), *kind)));
  }
  auto scalar = qop::ConstScalarOp::create(
      b, loc, scalarTy, b.getF64FloatAttr(coefficient), b.getF64FloatAttr(0.0));
  return qop::MakeProductOp::create(b, loc, productTy, scalar, factors);
}

// ---- Build a time-dependent drive term with callback ----
static Value buildDriveTerm(OpBuilder &b, Location loc, Value target,
                            StringRef spinKind, StringRef callbackName,
                            const WaveformCallbackData &callbackData) {
  auto handlerTy = qop::HandlerType::get(b.getContext());
  auto scalarTy = qop::ScalarType::get(b.getContext());
  auto productTy = qop::ProductType::get(b.getContext());

  auto kindAttr = qop::symbolizeHandlerKind(spinKind);
  if (!kindAttr)
    return {};
  auto spin =
      qop::SpinOp::create(b, loc, handlerTy, target,
                          qop::HandlerKindAttr::get(b.getContext(), *kindAttr));
  auto cbScalar = qop::CallbackScalarOp::create(
      b, loc, scalarTy, FlatSymbolRefAttr::get(b.getContext(), callbackName));
  cbScalar->setAttr("cudm.callback_kind",
                    b.getI32IntegerAttr(callbackData.kind));
  cbScalar->setAttr("cudm.callback_params",
                    b.getDenseF64ArrayAttr(callbackData.parameters));
  auto product =
      qop::MakeProductOp::create(b, loc, productTy, cbScalar, ValueRange{spin});
  return product;
}

struct PulseToQOpPass
    : public PassWrapper<PulseToQOpPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseToQOpPass)

  StringRef getArgument() const final { return "pulse-to-qop"; }
  StringRef getDescription() const final {
    return "Lower pulse dialect ops to qop Hamiltonian/Lindblad construction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<qop::QOpDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getContext());

    // Collect drive ops, qubit info, and dissipator metadata
    SmallVector<Operation *> driveOps;
    DenseMap<int64_t, double> qubitFreqHz;
    Operation *unsupportedMeasurement = nullptr;

    if (auto frequencies =
            module->getAttrOfType<DenseF64ArrayAttr>("pulse.qubit_freq_hz")) {
      for (auto [index, frequency] : llvm::enumerate(frequencies.asArrayRef()))
        qubitFreqHz[static_cast<int64_t>(index)] = frequency;
    }

    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "pulse.get_drive_line") {
        if (auto qubitAttr = op->getAttrOfType<IntegerAttr>("qubit")) {
          int64_t qi = qubitAttr.getInt();
          if (auto freqAttr = op->getAttrOfType<FloatAttr>("frequency_hz"))
            qubitFreqHz[qi] = freqAttr.getValueAsDouble();
        }
      }
      if (op->getName().getStringRef() == "pulse.drive")
        driveOps.push_back(op);
      if (op->getName().getStringRef() == "pulse.readout" ||
          op->getName().getStringRef() == "pulse.iq_acquire")
        unsupportedMeasurement = op;
    });
    if (unsupportedMeasurement) {
      unsupportedMeasurement->emitError(
          "measurement is not supported by cuDensityMat evolution; remove "
          "readout/acquisition operations before lowering");
      return signalPassFailure();
    }

    // Carry the scheduled pulse interval into the evolution lowering. Pulse
    // time is expressed in virtual clock units; cuDensityMat callbacks and
    // integrators use nanoseconds.
    int64_t endVtu = 0;
    module.walk([&](Operation *op) {
      auto start = op->getAttrOfType<IntegerAttr>("start_vtu");
      auto duration = op->getAttrOfType<IntegerAttr>("duration_vtu");
      if (start && duration)
        endVtu = std::max(endVtu, start.getInt() + duration.getInt());
    });
    double clockGHz = 1.0;
    if (auto clock = module->getAttrOfType<FloatAttr>("pulse.clock_ghz"))
      clockGHz = clock.getValueAsDouble();
    if (clockGHz <= 0.0) {
      module.emitError("pulse.clock_ghz must be positive");
      return signalPassFailure();
    }
    if (endVtu > 0) {
      const double endNs = static_cast<double>(endVtu) / clockGHz;
      if (!module->hasAttr("qop.t_start"))
        module->setAttr("qop.t_start", b.getF64FloatAttr(0.0));
      if (!module->hasAttr("qop.t_end"))
        module->setAttr("qop.t_end", b.getF64FloatAttr(endNs));
      if (!module->hasAttr("qop.num_steps"))
        module->setAttr("qop.num_steps",
                        b.getI64IntegerAttr(std::max<int64_t>(100, endVtu)));
      if (!module->hasAttr("qop.integrator"))
        module->setAttr("qop.integrator", b.getStringAttr("rk4"));
    }

    // Find the func.func @main and insert QOp construction at the end
    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }

    // Insert before the return op
    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();

    auto productTy = qop::ProductType::get(b.getContext());
    auto opTy = qop::OpType::get(b.getContext());
    auto superOpTy = qop::SuperOpType::get(b.getContext());

    SmallVector<Value> allProducts;

    // Evolution uses one rotating frame per qubit. Bare qubit frequencies are
    // therefore removed from H; tone detuning is encoded in each drive
    // callback below. This avoids a numerically stiff multi-GHz lab-frame ODE.
    auto appendInteractions = [&](StringRef pairAttrName,
                                  StringRef strengthAttrName,
                                  StringRef spinKind) -> LogicalResult {
      auto pairs = module->getAttrOfType<DenseI64ArrayAttr>(pairAttrName);
      auto strengths =
          module->getAttrOfType<DenseF64ArrayAttr>(strengthAttrName);
      if (!pairs && !strengths)
        return success();
      if (!pairs || !strengths || pairs.size() != 2 * strengths.size()) {
        module.emitError("interaction pair and strength metadata disagree for ")
            << pairAttrName;
        return failure();
      }
      for (auto [index, strength] : llvm::enumerate(strengths.asArrayRef())) {
        int64_t first = pairs[2 * index];
        int64_t second = pairs[2 * index + 1];
        if (first < 0 || second < 0 || !qubitFreqHz.count(first) ||
            !qubitFreqHz.count(second)) {
          module.emitError("interaction references an inactive qubit in ")
              << pairAttrName;
          return failure();
        }
        auto firstTarget =
            arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(first));
        auto secondTarget =
            arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(second));
        SmallVector<std::pair<Value, StringRef>> factors = {
            {firstTarget, spinKind}, {secondTarget, spinKind}};
        auto term =
            buildStaticProduct(b, loc, factors, strength * 2.0 * M_PI * 1.0e-9);
        if (term)
          allProducts.push_back(term);
      }
      return success();
    };
    if (failed(appendInteractions("pulse.coupling_pairs",
                                  "pulse.coupling_strength_hz", "spin_x")) ||
        failed(appendInteractions("pulse.crosstalk_pairs",
                                  "pulse.crosstalk_strength_hz", "spin_z")))
      return signalPassFailure();

    // 2. Time-dependent drive terms
    int driveIdx = 0;
    for (auto *op : driveOps) {
      std::string cbName = "drive_envelope_" + std::to_string(driveIdx);
      auto qubitIdx = traceQubit(op->getOperand(0));
      if (!qubitIdx) {
        op->emitError("cannot determine the physical qubit for this drive; "
                      "the line must originate from pulse.get_drive_line "
                      "with a 'qubit' attribute");
        return signalPassFailure();
      }

      auto target =
          arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(*qubitIdx));

      auto qubitFrequency = qubitFreqHz.find(*qubitIdx);
      if (qubitFrequency == qubitFreqHz.end()) {
        op->emitError("cannot determine the physical qubit frequency for "
                      "rotating-frame lowering");
        return signalPassFailure();
      }
      double driveScale = 1.0;
      if (auto scales = module->getAttrOfType<DenseF64ArrayAttr>(
              "pulse.drive_scale_rad_per_ns")) {
        if (*qubitIdx < 0 || static_cast<size_t>(*qubitIdx) >= scales.size() ||
            scales.asArrayRef()[*qubitIdx] <= 0.0) {
          op->emitError("drive amplitude scale is missing or non-positive");
          return signalPassFailure();
        }
        driveScale = scales.asArrayRef()[*qubitIdx];
      }

      auto callbackX = getWaveformCallbackData(
          op, clockGHz, qubitFrequency->second, driveScale, 0);
      auto callbackY = getWaveformCallbackData(
          op, clockGHz, qubitFrequency->second, driveScale, 1);
      if (!callbackX || !callbackY) {
        op->emitError("waveform cannot be lowered to a cuDensityMat callback; "
                      "schedule the program, specialize all numeric values, "
                      "and use a built-in or custom_samples waveform");
        return signalPassFailure();
      }

      // X-component
      auto termX =
          buildDriveTerm(b, loc, target, "spin_x", cbName + "_x", *callbackX);
      if (termX)
        allProducts.push_back(termX);

      // Y-component (for DRAG or nonzero phase)
      auto termY =
          buildDriveTerm(b, loc, target, "spin_y", cbName + "_y", *callbackY);
      if (termY)
        allProducts.push_back(termY);

      driveIdx++;
    }

    // 3. Assemble Hamiltonian: qop.make_sum
    Value hamiltonian;
    if (!allProducts.empty()) {
      hamiltonian = qop::MakeSumOp::create(b, loc, opTy, allProducts);
    } else {
      // Trivial Hamiltonian: identity
      auto target = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(0));
      auto term = buildStaticTerm(b, loc, target, "spin_i", 0.0, 0.0);
      hamiltonian = qop::MakeSumOp::create(b, loc, opTy, ValueRange{term});
    }

    // 4. Dissipators from module attributes (T1, T2)
    SmallVector<Value> collapseOps;

    auto t1AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
    if (!t1AttrRaw)
      t1AttrRaw = module->getAttrOfType<ArrayAttr>("t1_times");
    if (auto t1Attr = t1AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t1Attr.size(); qi++) {
        double t1 = cast<FloatAttr>(t1Attr[qi]).getValueAsDouble();
        if (t1 > 0) {
          double gamma = 1.0 / t1;
          auto target =
              arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(qi));
          auto lowering = buildStaticTerm(b, loc, target, "spin_lowering",
                                          std::sqrt(gamma), 0.0);
          if (lowering) {
            auto collapseOp =
                qop::MakeSumOp::create(b, loc, opTy, ValueRange{lowering});
            collapseOps.push_back(collapseOp);
          }
        }
      }
    }

    auto t2AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t2_times");
    if (!t2AttrRaw)
      t2AttrRaw = module->getAttrOfType<ArrayAttr>("t2_times");
    if (auto t2Attr = t2AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t2Attr.size(); qi++) {
        double t2 = cast<FloatAttr>(t2Attr[qi]).getValueAsDouble();
        double t1 = 0.0;
        auto t1a = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
        if (!t1a)
          t1a = module->getAttrOfType<ArrayAttr>("t1_times");
        if (t1a)
          if (qi < (int64_t)t1a.size())
            t1 = cast<FloatAttr>(t1a[qi]).getValueAsDouble();
        double gammaPhi = 0.0;
        if (t2 > 0) {
          gammaPhi = 1.0 / t2;
          if (t1 > 0)
            gammaPhi -= 1.0 / (2.0 * t1);
          if (gammaPhi < 0)
            gammaPhi = 0;
        }
        if (gammaPhi > 0) {
          auto target =
              arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(qi));
          auto dephase = buildStaticTerm(b, loc, target, "spin_z",
                                         std::sqrt(gammaPhi / 2.0), 0.0);
          if (dephase) {
            auto collapseOp =
                qop::MakeSumOp::create(b, loc, opTy, ValueRange{dephase});
            collapseOps.push_back(collapseOp);
          }
        }
      }
    }

    // 5. Construct Lindblad super-operator
    qop::LindbladOp::create(b, loc, superOpTy, hamiltonian, collapseOps);

    // The QOp graph above is self-contained. Consume the source Pulse graph so
    // the next conversion stage receives only QOp plus standard dialect ops.
    SmallVector<Operation *> pulseOps;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "pulse")
        pulseOps.push_back(op);
    });
    for (auto iter = pulseOps.rbegin(); iter != pulseOps.rend(); ++iter)
      (*iter)->erase();
  }
};

} // namespace

namespace pulse {

std::unique_ptr<mlir::Pass> createPulseToQOpPass() {
  return std::make_unique<PulseToQOpPass>();
}

} // namespace pulse
