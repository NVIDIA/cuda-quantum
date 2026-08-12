/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CircuitValidation.h"
#include "cudaq/Optimizer/Analysis/UnitaryBuilder.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <cmath>
#include <complex>
#include <optional>

using namespace mlir;

namespace cudaq::opt {

llvm::StringRef toString(DomainRejectionKind kind) {
  switch (kind) {
  case DomainRejectionKind::Measurement:
    return "measurement";
  case DomainRejectionKind::Reset:
    return "reset";
  case DomainRejectionKind::Noise:
    return "noise";
  case DomainRejectionKind::DynamicControlFlow:
    return "dynamic-control-flow";
  case DomainRejectionKind::UnsupportedCall:
    return "unsupported-call";
  case DomainRejectionKind::DynamicQubitRegister:
    return "dynamic-qubit-register";
  case DomainRejectionKind::TooManyQubits:
    return "too-many-qubits";
  case DomainRejectionKind::AncillaNotRestored:
    return "ancilla-not-restored";
  }
  return "unknown";
}

llvm::StringRef toString(EquivalenceGuarantee guarantee) {
  switch (guarantee) {
  case EquivalenceGuarantee::Exact:
    return "exact";
  case EquivalenceGuarantee::CleanAncilla:
    return "clean-ancilla";
  case EquivalenceGuarantee::BorrowedAncilla:
    return "borrowed-ancilla";
  }
  return "unknown";
}

/// Number of qubits denoted by a quantum type: `!quake.ref` and `!quake.wire`
/// are single qubits, `!quake.veq<N>` is N qubits, and a `!quake.struq` is the
/// sum over its members. Returns std::nullopt when the count is not statically
/// knowable (a dynamically-sized `!quake.veq`, directly or inside a `struq`)
/// and 0 for a non-quantum (classical) type.
static std::optional<std::size_t> qubitsInType(Type ty) {
  // Value (wire/control/cable) types always carry a static width.
  if (quake::isQuantumValueType(ty))
    return quake::getWireCount(ty);
  if (quake::isQuantumReferenceType(ty)) {
    if (!quake::isConstantQuantumRefType(ty))
      return std::nullopt;
    return quake::getAllocationSize(ty);
  }
  return 0;
}

/// True if op transfers control rather than falling through to the next
/// operation: a structured region construct (`cc.if`/`cc.loop`), an unwind out
/// of one, or a CFG branch (`cf.br`/`cf.cond_br`/`cf.switch`). The CFG cases
/// matter because an early return in a kernel reaches the validator as a
/// branch between blocks, not as a `cc.if`.
static bool isControlFlow(Operation *op) {
  return isa<cudaq::cc::IfOp, cudaq::cc::LoopOp, cudaq::cc::UnwindReturnOp,
             cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
             BranchOpInterface>(op);
}

namespace {
/// Accumulates rejections and the qubit tally while walking one kernel.
struct KernelChecker {
  BoundedUnitaryDomainStatus &status;
  StringRef kernel;
  unsigned exactQubitBound;
  std::size_t qubits = 0;
  std::size_t ancillaQubits = 0;
  bool sawDynamicRegister = false;
  bool sawControlFlow = false;

  void reject(DomainRejectionKind kind, Operation *op, std::string detail) {
    status.supported = false;
    status.rejections.push_back(
        {kind, kernel.str(), std::move(detail), op->getLoc()});
  }

  /// Add the qubits contributed by \p ty to the running tally, flagging a
  /// dynamic register at most once per kernel. Ancillas are counted twice on
  /// purpose: once in the total, which is what the bound applies to, and once
  /// on their own so the split can be reported.
  void tally(Type ty, Operation *op) {
    if (auto n = qubitsInType(ty)) {
      qubits += *n;
      if (quake::isAncilla(op))
        ancillaQubits += *n;
    } else if (!sawDynamicRegister) {
      sawDynamicRegister = true;
      reject(DomainRejectionKind::DynamicQubitRegister, op,
             "dynamically-sized !quake.veq");
    }
  }

  /// Report control flow at most once per kernel. An early return produces
  /// both a branch and extra blocks, and one diagnostic is enough.
  void rejectControlFlow(Operation *op, std::string detail) {
    if (sawControlFlow)
      return;
    sawControlFlow = true;
    reject(DomainRejectionKind::DynamicControlFlow, op, std::move(detail));
  }
};
} // namespace

BoundedUnitaryDomainStatus checkBoundedUnitaryDomain(ModuleOp module,
                                                     unsigned exactQubitBound) {
  BoundedUnitaryDomainStatus status;

  for (auto func : module.getOps<func::FuncOp>()) {
    // Declarations carry no body to validate.
    if (func.empty())
      continue;

    KernelChecker checker{status, func.getSymName(), exactQubitBound};

    // Qubits entering as kernel arguments count toward the bound.
    for (BlockArgument arg : func.getArguments())
      checker.tally(arg.getType(), func.getOperation());

    // A multi-block body is a CFG, which the unitary builder would flatten
    // into a straight line and get wrong. Catch it before looking at ops so
    // an unreachable or fallthrough-free block cannot slip by.
    if (!llvm::hasSingleElement(func.getBlocks()))
      checker.rejectControlFlow(func.getOperation(),
                                std::to_string(func.getBlocks().size()) +
                                    " blocks (CFG control flow)");

    func.walk([&](Operation *op) {
      // Structural checks that exclude a kernel, most specific first.
      if (isa<quake::MeasurementInterface>(op)) {
        checker.reject(DomainRejectionKind::Measurement, op,
                       op->getName().getStringRef().str());
      } else if (isa<quake::ResetOp>(op)) {
        checker.reject(DomainRejectionKind::Reset, op, "quake.reset");
      } else if (isa<quake::ApplyNoiseOp>(op)) {
        checker.reject(DomainRejectionKind::Noise, op, "quake.apply_noise");
      } else if (isControlFlow(op)) {
        checker.rejectControlFlow(op, op->getName().getStringRef().str());
      } else if (isa<CallOpInterface>(op)) {
        checker.reject(DomainRejectionKind::UnsupportedCall, op,
                       op->getName().getStringRef().str());
      }

      if (auto alloca = dyn_cast<quake::AllocaOp>(op))
        checker.tally(alloca.getResult().getType(), op);
      else if (isa<quake::BorrowWireOp, quake::NullWireOp>(op))
        checker.qubits += 1;
    });

    status.maxQubits = std::max(status.maxQubits, checker.qubits);
    status.maxAncillaQubits =
        std::max(status.maxAncillaQubits, checker.ancillaQubits);

    if (!checker.sawDynamicRegister && checker.qubits > exactQubitBound)
      checker.reject(DomainRejectionKind::TooManyQubits, func.getOperation(),
                     std::to_string(checker.qubits) + " > " +
                         std::to_string(exactQubitBound));
  }

  return status;
}

// Clifford domain preflight
llvm::StringRef toString(CliffordRejectionKind kind) {
  switch (kind) {
  case CliffordRejectionKind::Measurement:
    return "measurement";
  case CliffordRejectionKind::Reset:
    return "reset";
  case CliffordRejectionKind::Noise:
    return "noise";
  case CliffordRejectionKind::DynamicControlFlow:
    return "dynamic-control-flow";
  case CliffordRejectionKind::UnsupportedCall:
    return "unsupported-call";
  case CliffordRejectionKind::DynamicQubitRegister:
    return "dynamic-qubit-register";
  case CliffordRejectionKind::NonCliffordGate:
    return "non-clifford-gate";
  case CliffordRejectionKind::NonCliffordRotation:
    return "non-clifford-rotation";
  case CliffordRejectionKind::NonCliffordControl:
    return "non-clifford-control";
  }
  return "unknown";
}

static std::optional<double> constantAngle(Value value) {
  if (auto cst =
          dyn_cast_if_present<arith::ConstantFloatOp>(value.getDefiningOp()))
    return cast<FloatAttr>(cst.getValue()).getValueAsDouble();
  return std::nullopt;
}

static bool isMultipleOfHalfPi(double angle) {
  double k = angle / M_PI_2;
  return std::abs(k - std::round(k)) <= 1e-9;
}

static std::optional<std::size_t>
controlQubitCount(quake::OperatorInterface optor) {
  std::size_t count = 0;
  for (Value control : optor.getControls()) {
    if (auto n = qubitsInType(control.getType()))
      count += *n;
    else
      return std::nullopt;
  }
  return count;
}

static std::optional<CliffordRejectionKind>
classifyCliffordGate(quake::OperatorInterface optor) {
  Operation *op = optor.getOperation();

  auto controls = controlQubitCount(optor);
  if (!controls)
    return CliffordRejectionKind::NonCliffordControl;

  if (isa<quake::HOp, quake::SOp>(op)) {
    if (*controls != 0)
      return CliffordRejectionKind::NonCliffordControl;
    return std::nullopt;
  }
  if (isa<quake::XOp, quake::YOp, quake::ZOp>(op)) {
    if (*controls > 1)
      return CliffordRejectionKind::NonCliffordControl;
    return std::nullopt;
  }
  if (isa<quake::SwapOp>(op)) {
    if (*controls != 0)
      return CliffordRejectionKind::NonCliffordControl;
    return std::nullopt;
  }
  if (isa<quake::RxOp, quake::RyOp, quake::RzOp, quake::R1Op>(op)) {
    if (*controls != 0)
      return CliffordRejectionKind::NonCliffordControl;
    auto params = optor.getParameters();
    if (params.size() != 1)
      return CliffordRejectionKind::NonCliffordRotation;
    auto angle = constantAngle(params[0]);
    if (!angle || !isMultipleOfHalfPi(*angle))
      return CliffordRejectionKind::NonCliffordRotation;
    return std::nullopt;
  }
  return CliffordRejectionKind::NonCliffordGate;
}

CliffordDomainStatus checkCliffordDomain(ModuleOp module) {
  CliffordDomainStatus status;

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.empty())
      continue;

    StringRef kernel = func.getSymName();
    std::size_t qubits = 0;
    bool sawDynamicRegister = false;
    bool sawControlFlow = false;

    auto reject = [&](CliffordRejectionKind kind, Operation *op,
                      std::string detail) {
      status.supported = false;
      status.rejections.push_back(
          {kind, kernel.str(), std::move(detail), op->getLoc()});
    };

    auto tally = [&](Type ty, Operation *op) {
      if (auto n = qubitsInType(ty)) {
        qubits += *n;
      } else if (!sawDynamicRegister) {
        sawDynamicRegister = true;
        reject(CliffordRejectionKind::DynamicQubitRegister, op,
               "dynamically-sized !quake.veq");
      }
    };

    // See checkBoundedUnitaryDomain: control flow is reported once per kernel.
    auto rejectControlFlow = [&](Operation *op, std::string detail) {
      if (sawControlFlow)
        return;
      sawControlFlow = true;
      reject(CliffordRejectionKind::DynamicControlFlow, op, std::move(detail));
    };

    for (BlockArgument arg : func.getArguments())
      tally(arg.getType(), func.getOperation());

    if (!llvm::hasSingleElement(func.getBlocks()))
      rejectControlFlow(func.getOperation(),
                        std::to_string(func.getBlocks().size()) +
                            " blocks (CFG control flow)");

    func.walk([&](Operation *op) {
      if (isa<quake::MeasurementInterface>(op)) {
        reject(CliffordRejectionKind::Measurement, op,
               op->getName().getStringRef().str());
      } else if (isa<quake::ResetOp>(op)) {
        reject(CliffordRejectionKind::Reset, op, "quake.reset");
      } else if (isa<quake::ApplyNoiseOp>(op)) {
        reject(CliffordRejectionKind::Noise, op, "quake.apply_noise");
      } else if (isControlFlow(op)) {
        rejectControlFlow(op, op->getName().getStringRef().str());
      } else if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
        if (auto rejection = classifyCliffordGate(optor))
          reject(*rejection, op, op->getName().getStringRef().str());
      } else if (isa<CallOpInterface>(op)) {
        reject(CliffordRejectionKind::UnsupportedCall, op,
               op->getName().getStringRef().str());
      }

      if (auto alloca = dyn_cast<quake::AllocaOp>(op))
        tally(alloca.getResult().getType(), op);
      else if (isa<quake::BorrowWireOp, quake::NullWireOp>(op))
        qubits += 1;
    });

    status.maxQubits = std::max(status.maxQubits, qubits);
  }

  return status;
}

/// Build the dense unitary of func directly from the IR, reporting how many
/// ancillas were projected out and, on failure, whether they were dirty.
static LogicalResult computeKernelUnitary(func::FuncOp func,
                                          UnitaryBuilder::UMatrix &unitary,
                                          std::size_t &numAncillas,
                                          bool &dirtyAncilla) {
  UnitaryBuilder builder(unitary, /*upToMapping=*/false);
  auto status = builder.build(func);
  numAncillas = builder.getNumAncillas();
  dirtyAncilla = builder.sawDirtyAncilla();
  return status;
}

UnitaryComparisonResult compareUnitaries(func::FuncOp baseline,
                                         func::FuncOp candidate, double rtol,
                                         double atol) {
  UnitaryComparisonResult result;

  UnitaryBuilder::UMatrix baselineU;
  UnitaryBuilder::UMatrix candidateU;
  bool dirtyAncilla = false;
  if (failed(computeKernelUnitary(baseline, baselineU, result.baselineAncillas,
                                  dirtyAncilla))) {
    // A baseline that does not restore its own ancillas leaves nothing to
    // compare against, so it stays an error rather than a verdict.
    result.error = dirtyAncilla ? "baseline ancillas not restored to |0>"
                                : "failed to build baseline unitary";
    return result;
  }
  if (failed(computeKernelUnitary(candidate, candidateU,
                                  result.candidateAncillas, dirtyAncilla))) {
    if (!dirtyAncilla) {
      result.error = "failed to build candidate unitary";
      return result;
    }
    // The candidate is a well-formed circuit that answers the equivalence
    // question in the negative. It does not implement the baseline operator on
    // the system qubits, whatever the ancillas started as.
    result.computed = true;
    result.ancillaNotRestored = true;
    result.guarantee = EquivalenceGuarantee::CleanAncilla;
    return result;
  }
  if (result.baselineAncillas != 0 || result.candidateAncillas != 0)
    result.guarantee = EquivalenceGuarantee::CleanAncilla;
  if (baselineU.rows() != candidateU.rows() ||
      baselineU.cols() != candidateU.cols()) {
    result.error = "unitary dimension mismatch (different qubit counts)";
    return result;
  }

  result.computed = true;
  result.strictEqual = isApproxEqual(baselineU, candidateU,
                                     /*up_to_global_phase=*/false, rtol, atol);
  result.equalUpToGlobalPhase = isApproxEqual(
      baselineU, candidateU, /*up_to_global_phase=*/true, rtol, atol);

  if (result.equalUpToGlobalPhase) {
    // getGlobalPhaseConjugate(M) = exp(-i*arg(m0)) where m0 is the first
    // nonzero element of column 0. If normalizing both matrices makes them
    // equal, then candidate = exp(i*phase) * baseline with
    //   phase = arg(baseMult) - arg(candMult) = arg(baseMult * conj(candMult)).
    auto baseMult = getGlobalPhaseConjugate(baselineU, atol);
    auto candMult = getGlobalPhaseConjugate(candidateU, atol);
    result.phase = std::arg(baseMult * std::conj(candMult));
    result.phaseIsZero = std::abs(result.phase) <= atol;
  }

  return result;
}

} // namespace cudaq::opt
