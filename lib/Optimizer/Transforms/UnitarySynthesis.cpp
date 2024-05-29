/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcomplex.hpp>

#include <queue>

#define DEBUG_TYPE "unitary-synthesis"

namespace cudaq::opt {
#define GEN_PASS_DEF_UNITARYSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// This class implements the synthesis algorithm described in the following
/// paper - M. G. Davis, E. Smith, A. Tudor, K. Sen, I. Siddiqi and C. Iancu,
/// "Towards Optimal Topology Aware Quantum Circuit Synthesis," 2020 IEEE
/// International Conference on Quantum Computing and Engineering (QCE), Denver,
/// CO, USA, 2020, pp. 223-234, doi: 10.1109/QCE49297.2020.00036.

/// The gate-set of U3 and CNOT is universal for quantum computing, meaning that
/// any unitary matrix can be represented by a circuit consisting of only those
/// gates.
struct DefaultGateSet {};
struct U3 : public DefaultGateSet {
  double theta;
  double phi;
  double lambda;
  size_t qubit_idx;

  U3(size_t idx) {
    theta = 0.;
    phi = 0.;
    lambda = 0.;
    qubit_idx = idx;
  }
};

struct CNot : public DefaultGateSet {
  size_t control;
  size_t target;

  CNot(size_t c, size_t t) {
    control = c;
    target = t;
  }
};

/// TBD
class AStarSearchSynthesis {

  using Matrix = xt::xarray<std::complex<double>>;

  // One step in the circuit
  using Node = std::vector<DefaultGateSet>;

  Matrix targetUnitary;

  // an acceptability threshold
  double epsilon;

  // CNOT count limit δ
  size_t delta;

  ///  successor function s(n), takes a node as input and returns a list of
  ///  nodes  Given a node n as input, s(n) generates a successor by appending
  ///  to the  circuit structure described by n. It appends a CNOT followed by
  ///  two U3 gates.  One successor is generated for each possible placement of
  ///  the two-qubit gates  allowed by the given topology. The one-qubit gates
  ///  are placed immediately  after the CNOT, on the qubit lines that the CNOT
  ///  affects. A list of all  successors generated from n this way is returned.
  std::vector<Node> successorFunction(Node n);

  ///  optimization function p(n,Utarget), which takes a node and a unitary as
  ///  input  and returns a distance value The optimization function,
  ///  p(n,Utarget), is used  to find the closest matching circuit to a target
  ///  unitary given a circuit  structure. Given a node n and a unitary Utarget,
  ///  let U(n,x) represent the  unitary implemented by the circuit structure
  ///  represented by n when using the  vector x as parameters for the
  ///  parameterized gates in the circuit structure.  D(U(n,x),Utarget) is used
  ///  as an objective function, and is given to a  numerical optimizer, which
  ///  finds d = minxD(U(n,x),Utarget). The function  p(n,Utarget) returns d.

  double optimizationFunction(Node n);

  /// heuristic function employed by A*   h(n) = p(n,Utarget)∗9.3623/
  double heuristicFunction(double distance);

  /// distance function based on the Hilbert-Schmidt inner product
  /// D(U,Utarget) = 1− ⟨U, Utarget⟩HS / N = 1− Tr(U†Utarget)/N
  auto distanceMetric(Matrix current, Matrix target) {
    auto N = target.shape().size();
    auto HS = xt::sum(xt::conj(target) * current);
    auto distance = 1 - (HS / N);
    return distance;
  }

  Node buildRootNode() {
    Node root;
    size_t targetCount = std::log2(targetUnitary.shape().size());
    for (size_t q = 0; q < targetCount; q++) {
      root.emplace_back(U3(q));
    }
    return root;
  }

public:
  // constructor
  AStarSearchSynthesis(std::vector<std::complex<double>> m) {
    std::size_t dim = std::sqrt(m.size());
    std::vector<std::size_t> shape = {dim, dim};
    this->targetUnitary = xt::adapt(m, shape);
    this->epsilon = 1e-10;
    this->delta = 512;
  }

  /// Synthesis algorithm
  auto synthesize() {
    std::priority_queue<Node> pq;
    // n ← representation of U3 on each qubit
    Node root = buildRootNode();

    // push n onto queue with priority H(dbest)+0
    pq.push(root);

    // while queue is not empty do
    while (!pq.empty()) {
      // n ← pop from queue
      Node n = pq.top();
      pq.pop();
      // for all ni ∈S(n) do
      for (auto ni : successorFunction(n)) {
        // di ← P(ni, Utarget)
        auto di = optimizationFunction(ni);
        // if di < ε then
        if (di < epsilon) {
          return ni;
        }
        // if CNOT count of ni < δ then
        if (delta) {
          // push ni onto queue with priority H(di)+CNOT count of ni
          pq.push(ni);
        }
      }
    }
    Node empty;
    return empty;
  }
};

/// TBD
struct ReplaceUnitaryOp : public OpRewritePattern<quake::UnitaryOp> {
  using OpRewritePattern<quake::UnitaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    auto targets = op.getTargets();
    auto controls = op.getControls();
    Location loc = op->getLoc();

    auto constantUnitary = op.getConstantUnitary();
    if (constantUnitary.has_value()) {
      auto unitaryVal = constantUnitary.value();
      std::vector<std::complex<double>> targetUnitary(unitaryVal.size());
      for (std::size_t index = 0; auto &element : unitaryVal) {
        auto values = dyn_cast<DenseF32ArrayAttr>(element).asArrayRef();
        targetUnitary[index] = std::complex<double>(values[0], values[1]);
        index++;
      }

      /// TODO: Cache - maintain a map of unitary to decomposed gates and
      /// look-up first
      /// TODO: Add all known gates to the cache beforehand

      // Create an instance of the synthesis class, get "list" of replacement
      // operations If empty list, failed to synthesize, else, use
      // rewriter.create(...) calls to replace
      auto result = AStarSearchSynthesis(targetUnitary).synthesize();

      if (result.empty()) {
        op.emitOpError("failed to decompose the custom unitary");
      }
      /// TODO: Expand the replacer logic
      Value zero = rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat{0.0}, rewriter.getF64Type());
      std::array<Value, 3> parameters = {zero, zero, zero};
      for (auto t : targets) {
        rewriter.create<quake::U3Op>(loc, parameters, controls, t);
      }

    } else {
      op.emitOpError(
          "decomposition of parameterized custom operation not yet supported");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct UnitarySynthesisPass
    : public cudaq::opt::impl::UnitarySynthesisBase<UnitarySynthesisPass> {
  using UnitarySynthesisBase::UnitarySynthesisBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<ReplaceUnitaryOp>(context);
    ConversionTarget target(*context);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<quake::QuakeDialect>();
    target.addIllegalOp<quake::UnitaryOp>();

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op.emitOpError("could not replace unitary");
      signalPassFailure();
    }
  }
};
} // namespace