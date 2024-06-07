/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "common/EigenDense.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "cudaq/algorithms/optimizers/nlopt/nlopt.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <map>
#include <queue>
#include <random>
#include <unsupported/Eigen/KroneckerProduct>
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

using namespace std::complex_literals;

struct U3 {
private:
  double theta;
  double phi;
  double lambda;
  Eigen::Matrix<std::complex<double>, 2, 2> unitary;

public:
  size_t qubit_idx;

  U3(size_t idx) {
    theta = 0.;
    phi = 0.;
    lambda = 0.;
    qubit_idx = idx;
  }

  std::vector<double> getParams() { return {theta, phi, lambda}; }

  Eigen::Matrix<std::complex<double>, 2, 2> &
  getUnitary(const std::vector<double> &params) {
    theta = params[0];
    phi = params[1];
    lambda = params[2];

    unitary << std::cos(theta / 2.),
        -std::exp(lambda * 1i) * std::sin(theta / 2.),
        std::exp(phi * 1i) * std::sin(theta / 2.),
        std::exp(1i * (phi + lambda)) * std::cos(theta / 2.);

    return unitary;
  }
};

struct CNot {
  size_t control;
  size_t target;

private:
  Eigen::Matrix<std::complex<double>, 4, 4> unitary;

public:
  CNot(size_t c, size_t t) {
    control = c;
    target = t;
    unitary << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0;
  }

  Eigen::Matrix<std::complex<double>, 4, 4> getUnitary() { return unitary; }
};

struct Identity {
  size_t qubit_idx;
  Eigen::Matrix<std::complex<double>, 2, 2> unitary;

  Identity(size_t q) {
    qubit_idx = q;
    unitary = Eigen::Matrix<double, 2, 2>::Identity();
  }

  Eigen::Matrix<std::complex<double>, 2, 2> getUnitary() { return unitary; }
};
} // namespace DefaultGateSet

using Matrix =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;

/// One step (node) in the circuit (tree)
/// It can have either all U3 gates (one per qubit), or a CNOT gate followed by
/// two U3 gates Fill identity gate on unused qubit
// struct Node {
//   std::vector<std::variant<DefaultGateSet::U3, DefaultGateSet::CNot>>
//   elements; size_t qubitCount; size_t cnotCount; size_t paramCount; Matrix
//   matrix; bool visited;

//   // bool operator<(const Node &right) const {
//   //   return cnotCount < right.cnotCount;
//   // }

//   Node(size_t q) {
//     qubitCount = q;
//     cnotCount = 0;
//     paramCount = 0;
//     visited = false;
//   }

//   const Matrix &getMatrix() { return matrix; }

//   virtual const Matrix &computeMatrix(const std::vector<double> &params) {
//     return matrix;
//   }
// };

struct RootNode {

  std::vector<DefaultGateSet::U3 *> elements;
  size_t qubitCount;
  size_t cnotCount;
  size_t paramCount;
  Matrix matrix;
  bool visited;

  RootNode() {}

  RootNode(size_t q) {
    qubitCount = q;
    cnotCount = 0;
    paramCount = 0;
    visited = false;
    for (size_t i = 0; i < q; i++) {
      elements.emplace_back(new DefaultGateSet::U3(i));
      paramCount += 3;
    }
  }

  const Matrix &getMatrix() { return matrix; }

  const Matrix &computeMatrix(const std::vector<double> &params) {
    if (params.size() != paramCount) {
      llvm::report_fatal_error("incorrect number of parameters.");
    }
    size_t pIdx = 0;
    for (size_t i = 0; i < elements.size(); i++) {
      Matrix unitary;
      auto angles =
          std::vector<double>(params.begin() + pIdx, params.begin() + pIdx + 3);
      unitary = elements[i]->getUnitary(angles);
      pIdx += 3;
      if (0 == i) {
        matrix = unitary;
      } else {
        matrix = Eigen::kroneckerProduct(matrix, unitary).eval();
      }
    }
    visited = true;
    return matrix;
  }
};

struct CNotU3U3Node {

  std::vector<std::variant<DefaultGateSet::U3, DefaultGateSet::CNot>> elements;
  size_t qubitCount;
  size_t cnotCount;
  size_t paramCount;
  Matrix matrix;
  bool visited;

  std::vector<std::variant<DefaultGateSet::CNot, DefaultGateSet::Identity>>
      cNotCol;
  std::vector<std::variant<DefaultGateSet::U3, DefaultGateSet::Identity>> u3Col;

  CNotU3U3Node(size_t q, size_t c, size_t t) {

    qubitCount = q;
    visited = false;
    paramCount = 6;
    cnotCount = 1;
    bool cnotAdded = false;
    for (size_t i = 0; i < q; i++) {
      if ((i == c) || (i == t)) {
        if (!cnotAdded) {
          auto gate = DefaultGateSet::CNot(c, t);
          cNotCol.push_back(gate);
          cnotAdded = true;
        }
        auto gate = DefaultGateSet::U3(i);
        u3Col.push_back(gate);
      } else {
        cNotCol.emplace_back(DefaultGateSet::Identity(i));
        u3Col.emplace_back(DefaultGateSet::Identity(i));
      }
    }
  }

  const Matrix &getMatrix() { return matrix; }

  const Matrix &computeMatrix(const std::vector<double> &params) {

    if (params.size() != paramCount) {
      llvm::report_fatal_error("incorrect number of parameters.");
    }

    // Kronecker product of CNot column
    Matrix cnotColKP;
    for (size_t i = 0; i < cNotCol.size(); i++) {
      Matrix unitary;
      if (0 == cNotCol[i].index()) {
        auto &gate = std::get<DefaultGateSet::CNot>(cNotCol[i]);
        unitary = gate.getUnitary();
      } else if (0 == cNotCol[i].index()) {
        auto &gate = std::get<DefaultGateSet::Identity>(cNotCol[i]);
        unitary = gate.getUnitary();
      }
      if (i == 0) {
        cnotColKP = unitary;
      } else {
        cnotColKP = Eigen::kroneckerProduct(cnotColKP, unitary).eval();
      }
    }

    // Kronecker product of U3 column
    Matrix u3ColKP;
    size_t pIdx = 0;
    for (size_t i = 0; i < u3Col.size(); i++) {
      Matrix unitary;
      if (0 == u3Col[i].index()) {
        auto &gate = std::get<DefaultGateSet::U3>(u3Col[i]);
        auto angles = std::vector<double>(params.begin() + pIdx,
                                          params.begin() + pIdx + 3);
        unitary = gate.getUnitary(angles);
        pIdx += 3;
      } else if (0 == u3Col[i].index()) {
        auto &gate = std::get<DefaultGateSet::Identity>(u3Col[i]);
        unitary = gate.getUnitary();
      }
      if (0 == i) {
        u3ColKP = unitary;
      } else {
        u3ColKP = Eigen::kroneckerProduct(u3ColKP, unitary).eval();
      }
    }

    // Matrix multiplication of both columns
    matrix = cnotColKP * u3ColKP;

    visited = true;
    return matrix;
  }
};

// One path in the tree
struct Branch {
  RootNode root;
  std::vector<CNotU3U3Node> nodes;
  size_t cnotCount;
  size_t paramCount;
  Matrix matrix;

  bool operator<(const Branch &right) const {
    return cnotCount < right.cnotCount;
  }

  Branch(size_t q) {
    root = RootNode(q);
    cnotCount = 0;
    paramCount = root.paramCount;
  }

  void addNode(CNotU3U3Node &n) {
    nodes.emplace_back(n);
    // matrix = matrix * n.getMatrix();
    cnotCount += n.cnotCount;
    paramCount += n.paramCount;
  }

  bool isEmpty() { return nodes.empty(); }

  Matrix &getMatrix() { return matrix; }

  Matrix &computeMatrix(const std::vector<double> &params) {
    size_t pIdx = 0;
    auto pc = root.paramCount;
    auto angles =
        std::vector<double>(params.begin() + pIdx, params.begin() + pIdx + pc);
    matrix = root.computeMatrix(angles);
    pIdx += pc;
    for (size_t i = 0; i < nodes.size(); i++) {
      pc = nodes[i].paramCount;
      angles = std::vector<double>(params.begin() + pIdx,
                                   params.begin() + pIdx + pc);
      matrix = matrix * nodes[i].computeMatrix(angles);
      pIdx += pc;
    }
    return matrix;
  }
};

/// TBD
class AStarSearchSynthesis {

  Matrix targetUnitary;

  size_t dimension;

  size_t qubitCount;

  size_t cnotCounter;

  // an acceptability threshold
  double epsilon;

  // CNOT count limit δ
  size_t delta;

  std::vector<std::pair<size_t, size_t>> potentialCNots;

  ///  successor function s(n), takes a node as input and returns a list of
  ///  nodes  Given a node n as input, s(n) generates a successor by appending
  ///  to the  circuit structure described by n. It appends a CNOT followed by
  ///  two U3 gates.  One successor is generated for each possible placement of
  ///  the two-qubit gates  allowed by the given topology. The one-qubit gates
  ///  are placed immediately  after the CNOT, on the qubit lines that the CNOT
  ///  affects. A list of all  successors generated from n this way is returned.
  std::vector<CNotU3U3Node> successorFunction() {

    if (potentialCNots.empty()) {
      return {};
    }
    std::vector<CNotU3U3Node> successors;
    for (auto operands : potentialCNots) {
      /// TODO: Confirm qubit ordering
      successors.push_back(
          CNotU3U3Node(qubitCount, operands.second, operands.first));
    }
    return successors;
  }

  auto minimizeCost(Branch &b, const std::vector<double> &params) {
    cudaq::optimizers::cobyla optimizer;
    optimizer.initial_parameters = params;
    std::vector<double> lower_bounds(params.size(), 0.0);
    optimizer.lower_bounds = lower_bounds;
    std::vector<double> upper_bounds(params.size(), 2 * M_PI);
    optimizer.upper_bounds = upper_bounds;
    optimizer.max_eval = 100;
    optimizer.f_tol = epsilon;
    auto [opt_val, opt_params] = optimizer.optimize(
        params.size(), [&](const std::vector<double> &params) {
          auto f = distanceMetric(b.computeMatrix(params));
          return f;
        });
    return opt_params;
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
  double optimizationFunction(Branch b) {
    /// TODO: Choose initial parameters differently
    std::vector<double> params(b.paramCount, M_PI);
    params = minimizeCost(b, params);
    return distanceMetric(b.computeMatrix(params));
  }

  /// heuristic function employed by A*
  /// h(n) = p(n,Utarget)∗9.3623
  double heuristicFunction(double distance) {
    // dummy
    return 9.3623 * distance;
  }

  /// distance function based on the Hilbert-Schmidt inner product
  /// D(U,Utarget) = 1− ⟨U, Utarget⟩HS / N = 1− Tr(U†Utarget)/N
  double distanceMetric(const Matrix &current) {
    return std::abs(1 - std::abs((targetUnitary.conjugate() * current).sum()) /
                            qubitCount);
  }

public:
  // constructor
  AStarSearchSynthesis(std::vector<std::complex<double>> &m) {
    dimension = std::sqrt(m.size());
    qubitCount = std::log2(dimension);
    targetUnitary = Eigen::Map<Matrix>(m.data(), dimension, dimension);

    epsilon = 1e-8;
    cnotCounter = 0;

    for (size_t i = 0; i < qubitCount - 1; i++) {
      for (size_t j = i + 1; j < qubitCount; j++) {
        potentialCNots.emplace_back(std::make_pair(i, j));
        potentialCNots.emplace_back(std::make_pair(j, i));
      }
    }
    delta = potentialCNots.size() / 2;
  }

  /// Synthesis algorithm
  Branch synthesize() {

    std::priority_queue<Branch> pq;

    // n ← representation of U3 on each qubit
    // Node root = RootNode(qubitCount);
    Branch result(qubitCount);

    // push n onto queue with priority H(dbest)+0
    pq.push(result);

    // while queue is not empty do
    while (!pq.empty()) {
      // n ← pop from queue
      Branch b = pq.top();
      pq.pop();

      result = b;
      cnotCounter = b.cnotCount;

      // for all ni ∈S(n) do
      for (auto ni : successorFunction()) {
        Branch current = result;
        current.addNode(ni);

        // di ← P(ni, Utarget)
        auto di = optimizationFunction(current);
        // di < ε
        if (di < epsilon) {
          return result;
        }
        // if CNOT count of ni < δ then
        if (cnotCounter < delta) {
          // push ni onto queue with priority H(di)+CNOT count of ni
          pq.push(current);
        }
      }
    }

    optimizationFunction(result);
    return result;
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

      // // if (result.isEmpty()) {
      // //   op.emitOpError("failed to decompose the custom unitary");
      // // }

      /// TODO: Save across unitaries? Or can it be automatically optimized?
      // std::map<double, Value> paramValMap;

      for (auto &gate : result.root.elements) {
        std::vector<double> inParams = gate->getParams();
        std::array<Value, 3> outParams;
        for (size_t i = 0; i < inParams.size(); i++) {
          outParams[i] = rewriter.create<arith::ConstantFloatOp>(
              loc, APFloat{inParams[i]}, rewriter.getF64Type());
        }
        rewriter.create<quake::U3Op>(loc, outParams, controls,
                                     targets[gate->qubit_idx]);
      }

      for (auto nodes : result.nodes) {
        for (auto op : nodes.elements) {
          if (0 == op.index()) {
            // insert U3
            auto &gate = std::get<DefaultGateSet::U3>(op);
            std::vector<double> inParams = gate.getParams();
            std::array<Value, 3> outParams;
            for (size_t i = 0; i < inParams.size(); i++) {
              outParams[i] = rewriter.create<arith::ConstantFloatOp>(
                  loc, APFloat{inParams[i]}, rewriter.getF64Type());
            }
            rewriter.create<quake::U3Op>(loc, outParams, controls,
                                         targets[gate.qubit_idx]);
          } else if (1 == op.index()) {
            // insert cnot
            auto &gate = std::get<DefaultGateSet::CNot>(op);
            rewriter.create<quake::XOp>(loc, targets[gate.control],
                                        targets[gate.target]);
          }
        } // per element
      }   // per node

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