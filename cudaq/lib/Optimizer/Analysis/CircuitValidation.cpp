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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
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
  }
  return "unknown";
}

/// Number of qubits denoted by a quantum type: `!quake.ref` and `!quake.wire`
/// are single qubits. `!quake.veq<N>` is N qubits. Returns std::nullopt for a
/// dynamically-sized `!quake.veq` and 0 for a non-quantum (classical) type.
static std::optional<std::size_t> qubitsInType(Type ty) {
  if (isa<quake::RefType, quake::WireType>(ty))
    return 1;
  if (auto veq = dyn_cast<quake::VeqType>(ty)) {
    if (veq.hasSpecifiedSize())
      return veq.getSize();
    return std::nullopt;
  }
  return 0;
}

namespace {
/// Accumulates rejections and the qubit tally while walking one kernel.
struct KernelChecker {
  BoundedUnitaryDomainStatus &status;
  StringRef kernel;
  unsigned exactQubitBound;
  std::size_t qubits = 0;
  bool sawDynamicRegister = false;

  void reject(DomainRejectionKind kind, Operation *op, std::string detail) {
    status.supported = false;
    status.rejections.push_back(
        {kind, kernel.str(), std::move(detail), op->getLoc()});
  }

  /// Add the qubits contributed by \p ty to the running tally, flagging a
  /// dynamic register at most once per kernel.
  void tally(Type ty, Operation *op) {
    if (auto n = qubitsInType(ty)) {
      qubits += *n;
    } else if (isa<quake::VeqType>(ty) && !sawDynamicRegister) {
      sawDynamicRegister = true;
      reject(DomainRejectionKind::DynamicQubitRegister, op,
             "dynamically-sized !quake.veq");
    }
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

    func.walk([&](Operation *op) {
      // Structural checks that exclude a kernel, most specific first.
      if (isa<quake::MeasurementInterface>(op)) {
        checker.reject(DomainRejectionKind::Measurement, op,
                       op->getName().getStringRef().str());
      } else if (isa<quake::ResetOp>(op)) {
        checker.reject(DomainRejectionKind::Reset, op, "quake.reset");
      } else if (isa<quake::ApplyNoiseOp>(op)) {
        checker.reject(DomainRejectionKind::Noise, op, "quake.apply_noise");
      } else if (isa<cudaq::cc::IfOp, cudaq::cc::LoopOp>(op)) {
        checker.reject(DomainRejectionKind::DynamicControlFlow, op,
                       op->getName().getStringRef().str());
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

    if (!checker.sawDynamicRegister && checker.qubits > exactQubitBound)
      checker.reject(DomainRejectionKind::TooManyQubits, func.getOperation(),
                     std::to_string(checker.qubits) + " > " +
                         std::to_string(exactQubitBound));
  }

  return status;
}

/// Build the dense unitary of \p func directly from the IR.
static LogicalResult computeKernelUnitary(func::FuncOp func,
                                          UnitaryBuilder::UMatrix &unitary) {
  UnitaryBuilder builder(unitary, /*upToMapping=*/false);
  return builder.build(func);
}

UnitaryComparisonResult compareUnitaries(func::FuncOp baseline,
                                         func::FuncOp candidate, double rtol,
                                         double atol) {
  UnitaryComparisonResult result;

  UnitaryBuilder::UMatrix baselineU;
  UnitaryBuilder::UMatrix candidateU;
  if (failed(computeKernelUnitary(baseline, baselineU))) {
    result.error = "failed to build baseline unitary";
    return result;
  }
  if (failed(computeKernelUnitary(candidate, candidateU))) {
    result.error = "failed to build candidate unitary";
    return result;
  }
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
