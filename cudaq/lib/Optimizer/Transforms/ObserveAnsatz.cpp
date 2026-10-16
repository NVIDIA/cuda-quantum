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

/// A qubit marked by a compiler-generated `quake.evince`, identified by the
/// evince op and the position of the operand that names it.
using EvincedQubit = std::pair<Operation *, unsigned>;

/// Collect the qubits the kernel declared, in allocation order.
///
/// `inject-implicit-output` emits the evince operands in that order, and the
/// order survives the rest of the pipeline: the mapper threads evince as a
/// wire pass-through, so operand `i` still names the same qubit afterwards. A
/// qubit can be evinced more than once when the returns do not share an exit
/// block, so keep only the first occurrence of each.
void collectEvincedQubits(func::FuncOp funcOp,
                          SmallVectorImpl<EvincedQubit> &qubits) {
  SmallPtrSet<Value, 16> seen;
  funcOp.walk([&](cudaq::quake::EvinceOp evince) {
    for (auto [index, opnd] : llvm::enumerate(evince->getOperands()))
      if (cudaq::quake::isQuantumType(opnd.getType()) && seen.insert(opnd).second)
        qubits.emplace_back(evince.getOperation(), index);
  });
}

/// Append the basis change and measurement for one Pauli term in front of the
/// evince ops, mirroring what `add-measurements` does for `sample`.
///
/// Only the qubits in the term's support are measured: the expectation value
/// is the parity of the whole register, so measuring an identity position
/// would corrupt it.
LogicalResult appendMeasurements(func::FuncOp funcOp, ArrayRef<bool> bsf) {
  SmallVector<EvincedQubit> qubits;
  collectEvincedQubits(funcOp, qubits);

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

  std::size_t measureNum = 0;
  for (std::size_t i = 0; i < nQubits; ++i) {
    bool xElement = bsf[i];
    bool zElement = bsf[i + nQubits];
    MeasureBasis basis = MeasureBasis::I;
    if (xElement && zElement)
      basis = MeasureBasis::Y;
    else if (xElement)
      basis = MeasureBasis::X;
    // Nothing to do for Z or the identity.

    auto [evince, position] = qubits[i];
    auto loc = evince->getLoc();
    builder.setInsertionPoint(evince);

    // Note: in value semantics `appendMeasurement` threads the rotated wire
    // through, which rewrites this operand of the evince.
    Value qubit = evince->getOperand(position);
    appendMeasurement(basis, builder, loc, qubit);

    if (!xElement && !zElement)
      continue;

    char regName[16];
    std::snprintf(regName, sizeof(regName), "r%05lu", measureNum++);
    auto nameAttr = builder.getStringAttr(regName);
    auto ty = qubit.getType();
    if (cudaq::quake::isLinearType(ty)) {
      auto meas = cudaq::quake::MzOp::create(
          builder, loc, TypeRange{measTy, ty}, qubit, nameAttr);
      qubit.replaceAllUsesExcept(meas.getResult(1), meas);
    } else {
      auto resTy = isa<cudaq::quake::RefType>(ty) ? Type{measTy} : sequenceTy;
      cudaq::quake::MzOp::create(builder, loc, resTy, qubit, nameAttr);
    }
  }

  return success();
}

/// This pass appends the basis changes and `mz` operations for a single Pauli
/// term to an ansatz, in front of the `quake.evince` ops that mark the qubits
/// the kernel declared.
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

    // `observe` measures the whole register itself, so a kernel that measures
    // for itself cannot be observed.
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
