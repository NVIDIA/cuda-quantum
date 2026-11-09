/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_OBSERVEANSATZ
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Enum type for Paulis
enum class MeasureBasis { I, X, Y, Z };

/// Given an X or Y MeasureBasis, apply a Hadamard or $ry(\pi/2)$ respectively.
void appendMeasurement(MeasureBasis &basis, OpBuilder &builder, Location &loc,
                       Value &qubit) {
  SmallVector<Value> targets{qubit};
  if (cudaq::quake::isLinearType(qubit.getType())) {
    // Value semantics
    auto wireTy = cudaq::quake::WireType::get(builder.getContext());
    if (basis == MeasureBasis::X) {
      auto newOp = cudaq::quake::HOp::create(
          builder, loc, TypeRange{wireTy}, /*is_adj=*/false, ValueRange{},
          ValueRange{}, targets, DenseBoolArrayAttr{});
      qubit.replaceAllUsesExcept(newOp.getResult(0), newOp);
      qubit = newOp.getResult(0);
    } else if (basis == MeasureBasis::Y) {
      llvm::APFloat d(M_PI_2);
      Value rotation =
          arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(), d);
      auto newOp = cudaq::quake::RxOp::create(
          builder, loc, TypeRange{wireTy}, /*is_adj=*/false,
          ValueRange{rotation}, ValueRange{}, ValueRange{qubit},
          DenseBoolArrayAttr{});
      qubit.replaceAllUsesExcept(newOp.getResult(0), newOp);
      qubit = newOp.getResult(0);
    }
  } else {
    // Reference semantics
    if (basis == MeasureBasis::X) {
      cudaq::quake::HOp::create(builder, loc, ValueRange{}, targets);
    } else if (basis == MeasureBasis::Y) {
      llvm::APFloat d(M_PI_2);
      Value rotation =
          arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(), d);
      SmallVector<Value> params{rotation};
      cudaq::quake::RxOp::create(builder, loc, params, ValueRange{}, targets);
    }
  }
}

/// A qubit named by a `cudaq::record`: the evince op, the position of the
/// operand that names it, and its index when that operand is a whole veq.
using EvincedQubit =
    std::tuple<Operation *, unsigned, std::optional<std::size_t>>;

/// Collect the recorded qubits, in record order: spin operator index `i` is
/// the `i`-th recorded qubit. The order survives the rest of the pipeline: the
/// mapper threads evince as a wire pass-through, so operand `i` still names
/// the same qubit afterwards.
void collectEvincedQubits(func::FuncOp funcOp,
                          SmallVectorImpl<EvincedQubit> &qubits) {
  funcOp.walk([&](cudaq::quake::EvinceOp evince) {
    if (!evince.getUserRecord())
      return;
    for (auto [index, opnd] : llvm::enumerate(evince->getOperands())) {
      if (!cudaq::quake::isQuantumType(opnd.getType()))
        continue;
      if (auto size = cudaq::quake::getVeqSize(opnd))
        for (std::size_t i = 0; i < *size; ++i)
          qubits.emplace_back(evince.getOperation(), index, i);
      else
        qubits.emplace_back(evince.getOperation(), index, std::nullopt);
    }
  });
}

/// Append the basis change and measurement for one Pauli term in front of the
/// record.
///
/// Only the qubits in the term's support are measured: the expectation value
/// is the parity of the whole register, so measuring an identity position
/// would corrupt it.
LogicalResult appendMeasurements(func::FuncOp funcOp, ArrayRef<bool> bsf) {
  SmallVector<EvincedQubit> qubits;
  collectEvincedQubits(funcOp, qubits);
  // Nothing to measure in a function without a record.
  if (qubits.empty())
    return success();

  const std::size_t nQubits = bsf.size() / 2;
  if (nQubits > qubits.size())
    return funcOp.emitOpError(
        "Invalid number of binary-symplectic elements provided: " +
        std::to_string(bsf.size()) + ". Must provide at most 2 * NQubits = " +
        std::to_string(2 * qubits.size()));

  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  auto measTy = cudaq::quake::MeasureType::get(ctx);
  auto sequenceTy = cudaq::cc::SequenceType::get(measTy);

  for (std::size_t i = 0; i < nQubits; ++i) {
    bool xElement = bsf[i];
    bool zElement = bsf[i + nQubits];
    MeasureBasis basis = MeasureBasis::I;
    if (xElement && zElement)
      basis = MeasureBasis::Y;
    else if (xElement)
      basis = MeasureBasis::X;
    // Nothing to do for Z or the identity.

    auto [evince, position, element] = qubits[i];
    auto loc = evince->getLoc();
    builder.setInsertionPoint(evince);

    // Note: in value semantics `appendMeasurement` threads the rotated wire
    // through, which rewrites this operand of the evince.
    Value qubit = evince->getOperand(position);
    if (element)
      qubit = cudaq::quake::ExtractRefOp::create(builder, loc, qubit, *element);
    appendMeasurement(basis, builder, loc, qubit);

    if (!xElement && !zElement)
      continue;

    auto ty = qubit.getType();
    if (cudaq::quake::isLinearType(ty)) {
      auto meas = cudaq::quake::MzOp::create(builder, loc,
                                             TypeRange{measTy, ty}, qubit);
      qubit.replaceAllUsesExcept(meas.getResult(1), meas);
    } else {
      auto resTy = isa<cudaq::quake::RefType>(ty) ? Type{measTy} : sequenceTy;
      cudaq::quake::MzOp::create(builder, loc, resTy, qubit);
    }
  }

  cudaq::quake::eraseUserRecords(funcOp);
  return success();
}

/// This pass appends the basis changes and `mz` operations for a single Pauli
/// term to an ansatz, in front of the kernel's `cudaq::record`.
class ObserveAnsatzPass
    : public cudaq::opt::impl::ObserveAnsatzBase<ObserveAnsatzPass> {
public:
  using ObserveAnsatzBase::ObserveAnsatzBase;

  ObserveAnsatzPass(const std::vector<bool> &bsfData)
      : binarySymplecticForm{bsfData.begin(), bsfData.end()} {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.empty())
      return;

    // We can get the pauli term info from the MLIR command line parser, or from
    // a programmatic use of this pass
    if (binarySymplecticForm.empty())
      for (auto &b : termBSF)
        binarySymplecticForm.push_back(b);

    // `observe` adds its own measurements, so a kernel that measures cannot be
    // observed.
    bool hasMeasurement = false;
    funcOp.walk([&](cudaq::quake::MeasurementInterface) {
      hasMeasurement = true;
      return WalkResult::interrupt();
    });
    if (hasMeasurement) {
      funcOp.emitOpError("Cannot observe kernel with measures in it.");
      signalPassFailure();
      return;
    }

    if (failed(appendMeasurements(funcOp, binarySymplecticForm)))
      signalPassFailure();
  }

protected:
  SmallVector<bool> binarySymplecticForm;
};

} // namespace

std::unique_ptr<mlir::Pass>
cudaq::opt::createObserveAnsatzPass(const std::vector<bool> &packed) {
  return std::make_unique<ObserveAnsatzPass>(packed);
}
