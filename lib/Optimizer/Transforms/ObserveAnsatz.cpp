/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
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
  if (quake::isLinearType(qubit.getType())) {
    // Value semantics
    auto wireTy = quake::WireType::get(builder.getContext());
    if (basis == MeasureBasis::X) {
      auto newOp = builder.create<quake::HOp>(
          loc, TypeRange{wireTy}, /*is_adj=*/false, ValueRange{}, ValueRange{},
          targets, DenseBoolArrayAttr{});
      qubit.replaceAllUsesExcept(newOp.getResult(0), newOp);
      qubit = newOp.getResult(0);
    } else if (basis == MeasureBasis::Y) {
      llvm::APFloat d(M_PI_2);
      Value rotation =
          builder.create<arith::ConstantFloatOp>(loc, d, builder.getF64Type());
      auto newOp = builder.create<quake::RxOp>(
          loc, TypeRange{wireTy}, /*is_adj=*/false, ValueRange{rotation},
          ValueRange{}, ValueRange{qubit}, DenseBoolArrayAttr{});
      qubit.replaceAllUsesExcept(newOp.getResult(0), newOp);
      qubit = newOp.getResult(0);
    }
  } else {
    // Reference semantics
    if (basis == MeasureBasis::X) {
      builder.create<quake::HOp>(loc, ValueRange{}, targets);
    } else if (basis == MeasureBasis::Y) {
      llvm::APFloat d(M_PI_2);
      Value rotation =
          builder.create<arith::ConstantFloatOp>(loc, d, builder.getF64Type());
      SmallVector<Value> params{rotation};
      builder.create<quake::RxOp>(loc, params, ValueRange{}, targets);
    }
  }
}

/// Define a struct to hold the metadata we'll need.
struct AnsatzMetadata {
  /// Number of qubits is necessary to check the number of binary symplectic
  /// elements provided
  std::size_t nQubits = 0;

  /// Track pre-existing measurements and attempt to remove them during
  /// processing
  SmallVector<Operation *> measurements;

  /// Whether or not the mapping pass has been run
  bool mappingPassRan = false;

  /// Map qubit indices to their mlir Value
  DenseMap<std::size_t, Value> qubitValues;

  /// Map qubit indices to their mlir veq Value and index
  DenseMap<std::size_t, std::pair<Value, std::size_t>> qubitValuesIndexed;
};

/// Define a map type for Quake functions to their associated metadata
using AnsatzFunctionInfo = DenseMap<Operation *, AnsatzMetadata>;

/// This analysis pass will count the number of qubits used in the ansatz
/// function, then number of measure ops present, and it will map qubit indices
/// to their associated MLIR values.
///
/// At this point this Analysis assumes that canonicalization has been run,
/// and that quake-synth has been run or the Quake code does not have any
/// runtime-only-known information.
struct AnsatzFunctionAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnsatzFunctionAnalysis)

  AnsatzFunctionAnalysis(Operation *op) { performAnalysis(op); }

  const AnsatzFunctionInfo &getAnalysisInfo() const { return infoMap; }

private:
  // Scan the body of a function for ops that will be used for profiling.
  void performAnalysis(Operation *operation) {
    auto funcOp = dyn_cast<func::FuncOp>(operation);
    if (!funcOp)
      return;

    AnsatzMetadata data;

    funcOp->walk([&](quake::BorrowWireOp op) {
      Value wire = op.getResult();
      // Wires are linear types that must be used exactly once, so traverse
      // those uses until the end of the linear operators.
      // NOTE - if this is ever moved to other passes that have different use
      // cases than this one, then it needs to be updated to support ResetOp,
      // which is not an operator interface (I don't think).
      while (auto gate =
                 dyn_cast<quake::OperatorInterface>(*wire.getUsers().begin())) {
        std::size_t qopNum = 0;
        auto controls = gate.getControls();
        for (auto w : controls) {
          if (w == wire)
            break;
          else
            qopNum++;
        }
        if (qopNum >= controls.size()) {
          for (auto w : gate.getTargets()) {
            if (w == wire)
              break;
            else
              qopNum++;
          }
        }
        wire = gate.getWires()[qopNum];
        if (wire.getUsers().empty())
          break;
      }
      data.qubitValues.insert({data.nQubits++, wire});
      return WalkResult::advance();
    });

    // walk and find all quantum allocations
    auto walkResult = funcOp->walk([&](quake::AllocaOp op) {
      Value result = op.getResult();
      if (auto veq = dyn_cast<quake::VeqType>(result.getType())) {
        // Update data.nQubits here and store qubit info as indexes into
        // the veq, in case we don't encounter any ExtractRefOps for
        // them later.
        if (veq.hasSpecifiedSize()) {
          for (std::size_t i = 0; i < veq.getSize(); i++)
            data.qubitValuesIndexed.insert({data.nQubits++, {result, i}});
        } else
          return WalkResult::interrupt(); // this is an error condition
      } else {
        // single alloc is for a single qubit. Update data.qubitValues here
        // because ExtractRefOp `walk` won't find any ExtractRefOp for this.
        data.qubitValues.insert({data.nQubits++, result});
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      emitError(funcOp.getLoc(), "VeqType with unspecified size found");
      return; // no cleanup necessary because infoMap is unmodified
    }

    // NOTE: assumes canonicalization and cse have run.
    funcOp->walk([&](quake::ExtractRefOp op) {
      if (op.hasConstantIndex())
        data.qubitValues.insert({op.getConstantIndex(), op.getResult()});
    });

    // If mapping has moved qubits or introduced auxillary qubits, update the
    // analysis accordingly.
    if (auto mappingAttr =
            dyn_cast_if_present<ArrayAttr>(funcOp->getAttr("mapping_v2p"))) {
      // First populate mapping_v2p[].
      SmallVector<std::size_t> mapping_v2p(mappingAttr.size());
      std::transform(
          mappingAttr.begin(), mappingAttr.end(), mapping_v2p.begin(),
          [](Attribute attr) { return cast<IntegerAttr>(attr).getInt(); });

      // Next create newQubitValues[]
      DenseMap<std::size_t, Value> newQubitValues;
      for (auto [origIx, mappedIx] : llvm::enumerate(mapping_v2p))
        newQubitValues[origIx] = data.qubitValues[mappedIx];

      // Now replace the values in data
      data.nQubits = mapping_v2p.size();
      data.qubitValues = newQubitValues;
      data.mappingPassRan = true;
    }

    // Count all measures
    funcOp->walk([&](quake::MzOp op) { data.measurements.push_back(op); });

    infoMap.insert({operation, data});
  }

  AnsatzFunctionInfo infoMap;
};

/// This pass will compute ansatz analysis meta data and use that in a custom
/// rewrite pattern to append basis changes + mz operations to the ansatz quake
/// function.
class ObserveAnsatzPass
    : public cudaq::opt::impl::ObserveAnsatzBase<ObserveAnsatzPass> {
public:
  using ObserveAnsatzBase::ObserveAnsatzBase;

  ObserveAnsatzPass(const std::vector<bool> &bsfData)
      : binarySymplecticForm{bsfData.begin(), bsfData.end()} {}

  /// Perform sanity checks and remove existing measurements.
  LogicalResult performPreprocessing(func::FuncOp funcOp,
                                     const AnsatzFunctionInfo &infoMap) {
    // Use an Analysis to count the number of qubits.
    auto iter = infoMap.find(funcOp);
    if (iter == infoMap.end())
      return funcOp.emitOpError("Errors encountered in pass analysis");
    auto nQubits = iter->second.nQubits;

    if (nQubits < termBSF.size() / 2)
      return funcOp.emitOpError("Invalid number of binary-symplectic elements "
                                "provided: " +
                                std::to_string(termBSF.size()) +
                                ". Must provide at most 2 * NQubits = " +
                                std::to_string(2 * nQubits));

    // If the mapping pass was not run, we expect no pre-existing measurements.
    if (!iter->second.mappingPassRan && !iter->second.measurements.empty())
      return funcOp.emitOpError("Cannot observe kernel with measures in it.");

    // Attempt to remove measurements. Note that the mapping pass may add
    // measurements to kernels that don't contain any measurements. For observe
    // kernels, we remove them here since we are adding specific measurements
    // below. Note: each `op` in the list of measurements must be removed by the
    // end of this loop, otherwise the end result may be incorrect.
    for (auto *op : iter->second.measurements) {
      bool safeToRemove = [&]() {
        for (auto user : op->getUsers())
          if (!isa<quake::SinkOp, quake::ReturnWireOp>(user))
            return false;
        return true;
      }();
      if (!safeToRemove)
        return funcOp.emitOpError(
            "Cannot observe kernel with non dangling measurements.");

      for (auto result : op->getResults())
        if (quake::isLinearType(result.getType()))
          result.replaceAllUsesWith(op->getOperand(0));

      op->dropAllReferences();
      op->erase();
    }
    return success();
  }

  /// Append measurements in the correct basis.
  LogicalResult appendMeasurements(func::FuncOp funcOp,
                                   const AnsatzFunctionInfo &infoMap) {
    // Use an Analysis to count the number of qubits.
    auto iter = infoMap.find(funcOp);
    if (iter == infoMap.end())
      return funcOp.emitOpError("Errors encountered in pass analysis");

    // Set nQubits so we only measure the requested qubits.
    auto nQubits = binarySymplecticForm.size() / 2;

    // We want to insert after the last quantum operation. We must perform this
    // after erasing the measurements.
    OpBuilder builder = OpBuilder::atBlockTerminator(&funcOp.getBody().back());
    auto loc = funcOp.getBody().back().getTerminator()->getLoc();
    Operation *last = &funcOp.getBody().back().front();
    funcOp.walk([&](Operation *op) {
      if (dyn_cast<quake::OperatorInterface>(op) ||
          dyn_cast<quake::AllocaOp>(op))
        last = op;
    });
    builder.setInsertionPointAfter(last);

    // Loop over the binary-symplectic form provided and append
    // measurements as necessary.
    SmallVector<Value> qubitsToMeasure;
    for (std::size_t i = 0; i < binarySymplecticForm.size() / 2; i++) {
      bool xElement = binarySymplecticForm[i];
      bool zElement = binarySymplecticForm[i + nQubits];
      MeasureBasis basis = MeasureBasis::I;
      if (xElement && zElement)
        basis = MeasureBasis::Y;
      else if (xElement)
        basis = MeasureBasis::X;

      // do nothing for z or identities

      // get the qubit value
      Value qubitVal;
      auto seek = iter->second.qubitValues.find(i);
      if (seek == iter->second.qubitValues.end()) {
        auto seekIndexed = iter->second.qubitValuesIndexed.find(i);
        if (seekIndexed == iter->second.qubitValuesIndexed.end())
          return funcOp.emitError(
              "ObserveAnsatz could not find the qubit to measure: " +
              std::to_string(i));
        auto veqOp = seekIndexed->second.first;
        auto index = seekIndexed->second.second;
        auto extractRef =
            builder.create<quake::ExtractRefOp>(loc, veqOp, index);
        qubitVal = extractRef.getResult();
      } else {
        qubitVal = seek->second;
      }

      // append the measurement basis change ops
      // Note: when using value semantics, qubitVal will be updated to the new
      // wire here.
      appendMeasurement(basis, builder, loc, qubitVal);

      if (xElement + zElement != 0)
        qubitsToMeasure.push_back(qubitVal);
    }

    auto measTy = quake::MeasureType::get(builder.getContext());
    auto wireTy = quake::WireType::get(builder.getContext());
    for (auto &[measureNum, qubitToMeasure] :
         llvm::enumerate(qubitsToMeasure)) {
      // add the measure
      char regName[16];
      std::snprintf(regName, sizeof(regName), "r%05lu", measureNum);
      if (quake::isLinearType(qubitToMeasure.getType())) {
        auto newOp = builder.create<quake::MzOp>(
            loc, TypeRange{measTy, wireTy}, ValueRange{qubitToMeasure},
            builder.getStringAttr(regName));
        qubitToMeasure.replaceAllUsesExcept(newOp.getResult(1), newOp);
      } else {
        builder.create<quake::MzOp>(loc, measTy, qubitToMeasure,
                                    builder.getStringAttr(regName));
      }
    }

    return success();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.empty())
      return;

    // We can get the pauli term info from the MLIR command line parser, or from
    // a programmatic use of this pass
    if (binarySymplecticForm.empty())
      for (auto &b : termBSF)
        binarySymplecticForm.push_back(b);

    // Compute the analysis info
    const auto &analysis = getAnalysis<AnsatzFunctionAnalysis>();
    const auto &funcAnalysisInfo = analysis.getAnalysisInfo();

    // Perform sanity checks and remove measurements.
    if (failed(performPreprocessing(funcOp, funcAnalysisInfo))) {
      signalPassFailure();
      return;
    }

    // Now append the measurements.
    if (failed(appendMeasurements(funcOp, funcAnalysisInfo)))
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
