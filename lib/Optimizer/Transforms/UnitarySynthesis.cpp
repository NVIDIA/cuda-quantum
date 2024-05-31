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

#include <map>
#include <queue>
#include <utility>

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

/// Using the universal gate-set of U3 and CNOT as the default
namespace DefaultGateSet {
struct U3 {
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

struct CNot {
  size_t control;
  size_t target;

  CNot(size_t c, size_t t) {
    control = c;
    target = t;
  }
};
} // namespace DefaultGateSet

/// One step (node) in the circuit (tree)
struct Node {
  std::vector<std::variant<DefaultGateSet::U3, DefaultGateSet::CNot>> elements;
  size_t cnotCount;

  bool operator<(const Node &right) const {
    return cnotCount < right.cnotCount;
  }

  Node() { cnotCount = 0; }

  Node(DefaultGateSet::U3 op) {
    elements.emplace_back(op);
    cnotCount = 0;
  }

  Node(DefaultGateSet::CNot op) {
    elements.emplace_back(op);
    cnotCount = 1;
  }
};

/// TBD
class AStarSearchSynthesis {

  using Matrix = xt::xarray<std::complex<double>>;

  Matrix targetUnitary;

  size_t dimension;

  size_t qubitCount;

  size_t cnotCounter;

  // an acceptability threshold
  double epsilon;

  // CNOT count limit δ
  size_t delta;

  /// implement n choose c with c = 2
  /// TODO: Compute only once
  std::vector<std::pair<size_t, size_t>> computeCombinationPairs(size_t count) {
    std::vector<std::pair<size_t, size_t>> result;
    for (size_t i = 0; i < count - 1; i++) {
      for (size_t j = i + 1; j < count; j++) {
        result.emplace_back(std::make_pair(i, j));
      }
    }
    return result;
  }

  ///  successor function s(n), takes a node as input and returns a list of
  ///  nodes  Given a node n as input, s(n) generates a successor by appending
  ///  to the  circuit structure described by n. It appends a CNOT followed by
  ///  two U3 gates.  One successor is generated for each possible placement of
  ///  the two-qubit gates  allowed by the given topology. The one-qubit gates
  ///  are placed immediately  after the CNOT, on the qubit lines that the CNOT
  ///  affects. A list of all  successors generated from n this way is returned.
  std::vector<Node> successorFunction(Node n) {

    auto potentialCNots = computeCombinationPairs(qubitCount);

    if (potentialCNots.empty()) {
      return {n};
    }

    std::vector<Node> successors;

    for (auto couple : potentialCNots) {
      auto s = n;
      /// TODO: Confirm qubit ordering
      s.elements.emplace_back(
          DefaultGateSet::CNot(couple.first, couple.second));
      s.cnotCount++;
      s.elements.emplace_back(DefaultGateSet::U3(couple.first));
      s.elements.emplace_back(DefaultGateSet::U3(couple.second));
      successors.push_back(s);
    }

    return successors;
  }

  ///  optimization function p(n,Utarget), which takes a node and a unitary as
  ///  input  and returns a distance value. The optimization function,
  ///  p(n,Utarget), is used  to find the closest matching circuit to a target
  ///  unitary given a circuit  structure. Given a node n and a unitary Utarget,
  ///  let U(n,x) represent the  unitary implemented by the circuit structure
  ///  represented by n when using the  vector x as parameters for the
  ///  parameterized gates in the circuit structure.  D(U(n,x),Utarget) is used
  ///  as an objective function, and is given to a  numerical optimizer, which
  ///  finds d = minxD(U(n,x),Utarget). The function  p(n,Utarget) returns d.
  double optimizationFunction(Node n) {
    // dummy
    return epsilon;
  }

  /// heuristic function employed by A*
  /// h(n) = p(n,Utarget)∗9.3623
  double heuristicFunction(double distance) {
    // dummy
    return 9.3623;
  }

  /// distance function based on the Hilbert-Schmidt inner product
  /// D(U,Utarget) = 1− ⟨U, Utarget⟩HS / N = 1− Tr(U†Utarget)/N
  auto distanceMetric(Matrix current, Matrix target) {
    auto N = target.shape().size();
    auto HS = xt::sum(xt::conj(target) * current);
    auto distance = 1 - (HS / N);
    return distance;
  }

  /// Identity gate on each qubit
  Node buildRootNode() {
    Node root;
    root.cnotCount = 0;
    for (size_t q = 0; q < qubitCount; q++) {
      root.elements.emplace_back(DefaultGateSet::U3(q));
    }
    return root;
  }

public:
  // constructor
  AStarSearchSynthesis(std::vector<std::complex<double>> m) {
    dimension = std::sqrt(m.size());
    std::vector<std::size_t> shape = {dimension, dimension};
    targetUnitary = xt::adapt(m, shape);
    qubitCount = std::log2(targetUnitary.shape().size());
    epsilon = 1e-10;
    delta = 512;
    cnotCounter = 0;
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
      cnotCounter += n.cnotCount;
      // for all ni ∈S(n) do
      for (auto ni : successorFunction(n)) {
        // di ← P(ni, Utarget)
        auto di = optimizationFunction(ni);
        // if di < ε then
        if (di <= epsilon) {
          return ni;
        }
        // if CNOT count of ni < δ then
        if (cnotCounter < delta) {
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

      auto result = AStarSearchSynthesis(targetUnitary).synthesize();

      if (result.elements.empty()) {
        op.emitOpError("failed to decompose the custom unitary");
      }

      /// TODO: Save across unitaries? Or can it be automatically optimized?
      std::map<APFloat, Value> paramValMap;

      for (auto op : result.elements) {
        if (0 == op.index()) {
          // insert U3
          auto gate = std::get<DefaultGateSet::U3>(op);
          std::array<APFloat, 3> inParams = {
              APFloat{gate.theta}, APFloat{gate.phi}, APFloat{gate.lambda}};
          std::array<Value, 3> outParams;
          for (size_t i = 0; i < inParams.size(); i++) {
            if (paramValMap.find(inParams[i]) == paramValMap.end()) {
              Value val = rewriter.create<arith::ConstantFloatOp>(
                  loc, inParams[i], rewriter.getF64Type());
              paramValMap[inParams[i]] = val;
              outParams[i] = val;
            } else {
              outParams[i] = paramValMap[inParams[i]];
            }
          }

          rewriter.create<quake::U3Op>(loc, outParams, controls,
                                       targets[gate.qubit_idx]);
        } else if (1 == op.index()) {
          // insert cnot
          auto gate = std::get<DefaultGateSet::CNot>(op);
          rewriter.create<quake::XOp>(loc, targets[gate.control],
                                      targets[gate.target]);
        }
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