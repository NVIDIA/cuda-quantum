/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

/// @brief Enum type for Paulis
enum class MeasureBasis { I, X, Y, Z };

/// @brief Given an X or Y MeasureBasis, apply a Hadamard or ry(pi/2)
/// respectively
/// @param basis
/// @param builder
/// @param loc
/// @param qubit
void appendMeasurement(MeasureBasis &basis, OpBuilder &builder, Location &loc,
                       Value &qubit) {
  SmallVector<Value> targets{qubit};
  if (basis == MeasureBasis::X) {
    builder.create<quake::HOp>(loc, ValueRange{}, targets);
  } else if (basis == MeasureBasis::Y) {
    llvm::APFloat d(M_PI_2);
    Value rotation =
        builder.create<arith::ConstantFloatOp>(loc, d, builder.getF64Type());
    SmallVector<Value> params{rotation};
    builder.create<quake::RyOp>(loc, params, ValueRange{}, targets);
  }
}

/// @brief Define a struct to hold the metadata we'll need
struct AnsatzMetadata {
  /// @brief Number of qubits is necessary to check the number of
  /// binary symplectic elements provided
  std::size_t nQubits = 0;

  /// @brief Check that we have no measures
  std::size_t nMeasures = 0;

  /// @brief Map qubit indices to their mlir Value
  DenseMap<std::size_t, Value> qubitValues;
};

/// @brief Define a map type for Quake functions to their associated metadata
using AnsatzFunctionInfo = DenseMap<Operation *, AnsatzMetadata>;

/// @brief This analysis pass will count the number of qubits used
/// in the ansatz function, then number of measure ops present, and it will
/// map qubit indices to their associated MLIR values.
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

    // walk and find all quantum allocations
    funcOp->walk([&](quake::AllocaOp op) {
      data.nQubits += op.getResult().getType().cast<quake::VeqType>().getSize();
    });

    // NOTE: assumes canonicalization and cse have run.
    funcOp->walk([&](quake::ExtractRefOp op) {
      if (op.hasConstantIndex())
        data.qubitValues.insert({op.getConstantIndex(), op.getResult()});
    });

    // Count all measures
    funcOp->walk([&](quake::MzOp op) { data.nMeasures++; });

    infoMap.insert({operation, data});
  }

  AnsatzFunctionInfo infoMap;
};

/// @brief This OpRewritePattern will use the quake ansatz analysis
/// info to append measurement basis change operations.
struct AppendMeasurements : public OpRewritePattern<func::FuncOp> {
  explicit AppendMeasurements(MLIRContext *ctx, const AnsatzFunctionInfo &info,
                              std::vector<bool> &bsf)
      : OpRewritePattern(ctx), infoMap(info), termBSF(bsf) {}

  /// @brief The pre-computed analysis information
  AnsatzFunctionInfo infoMap;

  /// @brief The Pauli term representation
  std::vector<bool> &termBSF;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(funcOp);

    OpBuilder builder = OpBuilder::atBlockTerminator(&funcOp.getBody().back());
    auto loc = funcOp.getBody().back().getTerminator()->getLoc();

    // We want to insert after the last quantum operation
    Operation *last = &funcOp.getBody().back().front();
    funcOp.walk([&](Operation *op) {
      if (dyn_cast<quake::OperatorInterface>(op))
        last = op;
    });
    builder.setInsertionPointAfter(last);

    // Use an Analysis to count the number of qubits.
    auto iter = infoMap.find(funcOp);
    assert(iter != infoMap.end());
    auto nQubits = iter->second.nQubits;
    auto nMeasures = iter->second.nMeasures;

    if (nQubits != termBSF.size() / 2) {
      std::string msg = "Invalid number of binary-symplectic elements "
                        "provided. Must provide 2 * NQubits = " +
                        std::to_string(2 * nQubits) + "\n";
      funcOp.emitError(msg);
      return failure();
    }

    if (nMeasures != 0) {
      std::string msg = "Cannot observe kernel with measures in it.\n";
      funcOp.emitError(msg);
      return failure();
    }

    // Loop over the binary-symplectic form provided and append
    // measurements as necessary.
    std::vector<Value> qubitsToMeasure;
    for (std::size_t i = 0; i < termBSF.size() / 2; i++) {
      bool xElement = termBSF[i];
      bool zElement = termBSF[i + nQubits];
      MeasureBasis basis = MeasureBasis::I;
      if (xElement && zElement)
        basis = MeasureBasis::Y;
      else if (xElement)
        basis = MeasureBasis::X;

      // do nothing for z or identities

      // get the qubit value
      auto seek = iter->second.qubitValues.find(i);
      if (seek == iter->second.qubitValues.end())
        continue;
      auto qubitVal = seek->second;

      // append the measurement basis change ops
      appendMeasurement(basis, builder, loc, qubitVal);

      if (xElement + zElement != 0)
        qubitsToMeasure.push_back(qubitVal);
    }

    for (auto &qubitToMeasure : qubitsToMeasure) {
      // add the measure
      builder.create<quake::MzOp>(loc, builder.getI1Type(), qubitToMeasure);
    }

    rewriter.finalizeRootUpdate(funcOp);
    return success();
  }
};

/// @brief This pass will compute ansatz analysis meta data and use that
/// in a custom rewrite pattern to append basis changes + mz operations
/// to the ansatz quake function.
class QuakeObserveAnsatzPass
    : public cudaq::opt::QuakeObserveAnsatzBase<QuakeObserveAnsatzPass> {
protected:
  std::vector<bool> binarySymplecticForm;

public:
  QuakeObserveAnsatzPass() = default;
  QuakeObserveAnsatzPass(std::vector<bool> &bsfData)
      : binarySymplecticForm(bsfData) {}

  void runOnOperation() override {
    auto funcOp = dyn_cast<func::FuncOp>(getOperation());
    if (!funcOp || funcOp.empty())
      return;

    // We can get the pauli term info from the MLIR
    // command line parser, or from a programmatic use of this pass
    if (binarySymplecticForm.empty())
      for (auto &b : termBSF)
        binarySymplecticForm.push_back(b);

    auto *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // Compute the analysis info
    const auto &analysis = getAnalysis<AnsatzFunctionAnalysis>();
    const auto &funcAnalysisInfo = analysis.getAnalysisInfo();
    patterns.insert<AppendMeasurements>(ctx, funcAnalysisInfo,
                                        binarySymplecticForm);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      emitError(funcOp.getLoc(), "failed to observe ansatz");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeObserveAnsatzPass() {
  return std::make_unique<QuakeObserveAnsatzPass>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createQuakeObserveAnsatzPass(std::vector<bool> &bsfData) {
  return std::make_unique<QuakeObserveAnsatzPass>(bsfData);
}
