/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/GraphPartioner.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace cudaq;

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKECIRCUITCUT
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

namespace {

bool isUsedAsControl(OpResult opResult, OperandRange operands) {
  if (operands.size() == 1)
    return false;

  if (opResult == operands.back())
    return false;

  return true;
}

/// @brief Given a Quake FuncOp, map it to a Graph.
void createGraph(func::FuncOp &quakeFunc, Graph &graph) {

  // Track Operation* to the GraphNode representing it
  std::map<Operation *, GraphNode> mapper;

  // Track node Ids.
  std::size_t id = 0;

  // First add |0> qubit nodes to start the graph,
  // and then add all quantum operations as nodes
  quakeFunc.walk([&](Operation *op) {
    if (isa<quake::NullWireOp, quake::OperatorInterface, quake::MxOp,
            quake::MyOp, quake::MzOp>(op)) {
      GraphNode node{op, id++};
      graph.insert({node, std::vector<GraphNode>{}});

      // Also build a map so we can go from Operation* to GraphNode.
      mapper.insert({op, node});
    }
    return WalkResult::advance();
  });

  // Add the edges now
  quakeFunc.walk([&](Operation *op) {
    if (isa<quake::NullWireOp, quake::OperatorInterface>(op)) {
      auto &thisNode = mapper.at(op);
      auto users = op->getUsers();
      for (auto user : users)
        if (isUsedAsControl(user->getResult(0),
                            mapper.at(user).op->getOperands()))
          // keep control nodes at the beginning of the vector
          graph[thisNode].insert(graph[thisNode].begin(), mapper.at(user));
        else
          graph[thisNode].emplace_back(mapper.at(user));
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "\n\nGraph data:\n");
  dumpGraph(graph);
}

/// @brief For each partitioned graph, add the requisite
/// Measure and State Prep nodes.
void addMeasureAndStatePrepNodes(std::size_t &nodeId,
                                 std::vector<Graph> &graphs) {

  // Loop through these partitions, look for graphs with
  // dangling edge connections. For the graph with an edge out to
  // a node in different graph, we need to add a MeasureNode. At the
  // same time we need to insert a StatePrep node at the other end of the
  // cut. The goal here is to create new vector of Graphs
  for (std::size_t i = 0; auto &g : graphs) {

    // loop over each nodes edges, if we find
    // and edge that is not in this partition, then
    // we will append a measure node. Then we should go
    // to that same node in the other partition, and
    // prepend a state prep node
    for (auto &[node, edges] : g) {
      for (std::size_t k = 0; k < edges.size(); k++) {
        auto &edge = edges[k];
        // Is the node that this node is connected to in this Graph?
        if (g.find(edge) == g.end()) {
          LLVM_DEBUG(llvm::dbgs() << "Graph " << i << " Edge " << edge.getName()
                                  << " not in this graph.\n");

          // The edge that this node connects to is not in this graph.
          // So we need to do 2 things: replace this edge node with a
          // Measure Node, and find the graph that the edge node is in and
          // add a StatePrepNode to it that leads into the edge node.

          // First add the measure node
          GraphNode mNode("MeasureNode", nodeId++);
          g.insert({mNode, std::vector<GraphNode>{}});

          // Must be the first node, it is not possible
          // for the 0th node to be in this graph
          GraphNode statePrep("StatePrepNode", 0);

          // What graph is edgeIter in?
          for (std::size_t j = 0; j < graphs.size(); j++) {
            if (j == i)
              continue;

            // Find the graph and insert the state prep node
            if (graphs[j].count(edge)) {
              graphs[j].insert({statePrep, std::vector<GraphNode>{edge}});
              break;
            }
          }

          // Replace this edge with the measure node
          edge = mNode;
        }
      }
    }

    i++;
  }
}

/// @brief Convert the input Graph to a Quake FuncOp and append to the
/// given ModuleOp.
void graphToQuake(ModuleOp &moduleOp, Graph &graph, Location loc,
                  const std::string &partitionFunctionName) {
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
  auto *ctx = moduleOp.getContext();

  // create the new Function for the current Sub-Graph
  auto newFunc = builder.create<func::FuncOp>(loc, partitionFunctionName,
                                              FunctionType::get(ctx, {}, {}));

  auto *rewriteEntryBlock = newFunc.addEntryBlock();
  builder.setInsertionPointToStart(rewriteEntryBlock);

  // Poor mans topo-sort, sort the nodes
  std::vector<GraphNode> nodes;
  sortNodes(graph, nodes);

  // Construct a map that takes GraphNode IDs to a vector
  // of input target / control Values.
  std::map<Operation *, std::vector<Value>> nodeInputControlValues;
  std::map<Operation *, std::vector<Value>> nodeInputTargetValues;

  // Lambda for inserting into the control target maps.
  auto checkAndInsert = [](std::map<Operation *, std::vector<Value>> &input,
                           Operation *op, Value value) {
    auto iter = input.find(op);
    if (iter == input.end())
      input.insert({op, std::vector<Value>{value}});
    else
      iter->second.push_back(value);
  };

  // Loop over the sorted Graph Nodes.
  for (auto &node : nodes) {
    // Handle Measure Node
    if (!node.op && node.getName().find("MeasureNode") != std::string::npos) {
      auto input = nodeInputTargetValues[node.op];
      builder.create<quake::CutMeasureOp>(loc, input[0]);
      continue;
    }

    // Handle State Prep Node
    if (!node.op && node.getName().find("StatePrepNode") != std::string::npos) {

      Value qubit = builder.create<quake::CutStatePrepOp>(
          loc, quake::WireType::get(ctx),
          builder.create<quake::NullWireOp>(loc, quake::WireType::get(ctx)));

      for (auto &edge : graph[node]) {
        // FIXME, need to figure out if StatePrepNode result is
        // connected as a control or target, this information can only
        // come from the corresponding cut_measure node on the other graph.
        checkAndInsert(nodeInputTargetValues, edge.op, qubit);
      }
      continue;
    }

    // Handle the NullWireOp qubit creation
    if (isa<quake::NullWireOp>(node.op)) {
      // Create the new NullWireOp, get its return Value
      Value newQubit =
          builder.create<quake::NullWireOp>(loc, quake::WireType::get(ctx));

      // Now we loop through all connections to this node and
      // look and see if this Value is used as a Control or a Target, and
      // add it to the appropriate map.
      for (auto &edge : graph[node]) {
        bool isControl = edge.op == nullptr
                             ? false
                             : isUsedAsControl(node.op->getResult(0),
                                               edge.op->getOperands());
        if (isControl)
          checkAndInsert(nodeInputControlValues, edge.op, newQubit);
        else
          checkAndInsert(nodeInputTargetValues, edge.op, newQubit);
      }

      continue;
    }

    if (isa<quake::OperatorInterface>(node.op)) {
      auto cloned = node.op->clone();
      std::vector<Value> operands;
      if (isa<quake::RxOp, quake::RyOp, quake::RzOp>(node.op)) {
        auto arithConstantOp =
            node.op->getOperand(0).getDefiningOp<arith::ConstantOp>().clone();
        operands.emplace_back(arithConstantOp);
        builder.insert(arithConstantOp);
      }

      for (auto &v : nodeInputControlValues[node.op])
        operands.push_back(v);

      for (auto &v : nodeInputTargetValues[node.op])
        operands.push_back(v);

      cloned->setOperands(operands);
      auto result = cloned->getResult(0);
      for (auto &edge : graph[node]) {
        bool isControl = edge.op == nullptr
                             ? false
                             : isUsedAsControl(node.op->getResult(0),
                                               edge.op->getOperands());
        if (isControl)
          checkAndInsert(nodeInputControlValues, edge.op, result);
        else
          checkAndInsert(nodeInputTargetValues, edge.op, result);
      }

      builder.insert(cloned);
      continue;
    }

    if (isa<quake::MxOp, quake::MyOp, quake::MzOp>(node.op)) {
      auto cloned = node.op->clone();
      cloned->setOperands(nodeInputTargetValues[node.op]);
      builder.insert(cloned);
      continue;
    }
  }

  builder.create<func::ReturnOp>(loc);

  return;
}

class QuakeCircuitCutPass
    : public cudaq::opt::QuakeCircuitCutBase<QuakeCircuitCutPass> {
public:
  using QuakeCircuitCutBase::QuakeCircuitCutBase;

  QuakeCircuitCutPass(std::size_t n) { numPartitions = n; }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // assert we only have 1 FuncOp.
    func::FuncOp quakeFunc;
    std::size_t numFuncs = 0;
    moduleOp.walk([&](func::FuncOp func) {
      if (func.isDeclaration())
        return WalkResult::advance();
      numFuncs++;
      quakeFunc = func;
      return WalkResult::advance();
    });

    if (numFuncs != 1) {
      moduleOp->emitOpError("Quake Circuit Cutting can only handle ModuleOps "
                            "with a single Quake FuncOp.");
      signalPassFailure();
      return;
    }

    // Don't run this if numPartitions == 1
    if (numPartitions < 2) {
      moduleOp.emitWarning(
          "Invalid number of partitions, must be greater than 1.");
      signalPassFailure();
      return;
    }

    // Assert the FuncOp is in the value semantic model
    auto result = quakeFunc.walk([](Operation *op) {
      if (auto qOp = dyn_cast<quake::OperatorInterface>(op)) {
        for (auto control : qOp.getControls())
          if (isa<quake::RefType>(control.getType()))
            return WalkResult::interrupt();
        for (auto target : qOp.getTargets())
          if (isa<quake::RefType>(target.getType()))
            return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      quakeFunc.emitError(
          "Quake Func must be in value-semantic form. Run memtoreg.");
      signalPassFailure();
      return;
    }

    // Disallow more cuts than qubits
    std::size_t numQubits = 0;
    quakeFunc.walk([&numQubits](quake::NullWireOp wire) { numQubits++; });
    if (numPartitions > numQubits) {
      moduleOp.emitWarning("Invalid number of partitions, must be less than or "
                           "equal to number of qubits.");
      signalPassFailure();
      return;
    }

    // Map Quake to a Graph
    Graph graph;
    createGraph(quakeFunc, graph);

    // Get the desired partitioner
    auto partitioner =
        cudaq::registry::get<cudaq::GraphPartitioner>(graphPartitioner);
    if (!partitioner) {
      moduleOp.emitError("Invalid partitioner (" + graphPartitioner +
                         "), cannot perform Quake Circuit Cutting.");
      signalPassFailure();
      return;
    }

    // Partition the graph
    auto graphs = partitioner->partition(graph, numPartitions);

    // Append Measure and StatePrep nodes
    std::size_t nodeId = graph.size();
    addMeasureAndStatePrepNodes(nodeId, graphs);

    LLVM_DEBUG(llvm::dbgs() << "\nMeasure/StatePrep Graphs:\n");
    for (auto &g : graphs) {
      LLVM_DEBUG(llvm::dbgs() << "\nGraph data:\n");
      dumpGraph(g);
    }

    // For each graph, order the nodes and re-number from 0,..,N
    for (std::size_t i = 0; auto &g : graphs)
      graphToQuake(moduleOp, g, quakeFunc.getLoc(),
                   quakeFunc.getName().str() + ".partition_" +
                       std::to_string(i++));
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createQuakeCircuitCutPass() {
  return std::make_unique<QuakeCircuitCutPass>();
}

std::unique_ptr<Pass> cudaq::opt::createQuakeCircuitCutPass(std::size_t n) {
  return std::make_unique<QuakeCircuitCutPass>(n);
}
