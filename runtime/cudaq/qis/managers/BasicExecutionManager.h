/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "cudaq/qis/execution_manager.h"

#include <complex>
#include <functional>
#include <map>
#include <queue>
#include <stack>

namespace cudaq {

/// @brief The `BasicExecutionManager` provides a common base class for
/// specializations that implement the `ExecutionManager` type. Most of the
/// required `ExecutionManager` functionality is implemented here, with
/// backend-execution-specific details left for further subtypes. This type
/// enqueues all quantum operations and flushes them at specific synchronization
/// points. Subtypes should implement concrete operation execution, qudit
/// measurement, allocation, and deallocation, and execution context handling
/// (e.g. sampling)
class BasicExecutionManager : public cudaq::ExecutionManager {
protected:
  /// @brief An instruction is composed of a operation name,
  /// a optional set of rotation parameters, control qudits,
  /// target qudits, and an optional spin_op.
  using Instruction = std::tuple<std::string, std::vector<double>,
                                 std::vector<cudaq::QuditInfo>,
                                 std::vector<cudaq::QuditInfo>, spin_op>;

  /// @brief `typedef` for a queue of instructions
  using InstructionQueue = std::vector<Instruction>;

  /// @brief When adjoint operations are requested we can store them here for
  /// delayed execution
  std::vector<InstructionQueue> adjointQueueStack;

  /// @brief When we are in a control region, we need to store extra control
  /// qudit ids.
  std::vector<std::size_t> extraControlIds;

  /// @brief Subtype-specific method for affecting the execution of a queued
  /// instruction.
  virtual void executeInstruction(const Instruction &inst) = 0;

public:
  BasicExecutionManager() = default;
  virtual ~BasicExecutionManager() = default;

  void startAdjointRegion() override { adjointQueueStack.emplace_back(); }

  void endAdjointRegion() override {
    assert(!adjointQueueStack.empty() && "There must be at least one queue");

    auto adjointQueue = std::move(adjointQueueStack.back());
    adjointQueueStack.pop_back();

    std::reverse(adjointQueue.begin(), adjointQueue.end());

    // Select the queue to which these instructions will be added.
    if (!adjointQueueStack.empty()) {
      for (auto &&instruction : adjointQueue)
        adjointQueueStack.back().push_back(instruction);
      return;
    }

    for (auto &instruction : adjointQueue)
      executeInstruction(instruction);
  }

  void startCtrlRegion(const std::vector<std::size_t> &controls) override {
    for (auto c : controls)
      extraControlIds.push_back(c);
  }

  void endCtrlRegion(const std::size_t n_controls) override {
    extraControlIds.resize(extraControlIds.size() - n_controls);
  }

  /// The goal for apply is to create a new element of the
  /// instruction queue (a tuple).
  void apply(std::string gateName, std::vector<double> params,
             std::vector<cudaq::QuditInfo> controls,
             const std::vector<cudaq::QuditInfo> &targets,
             bool isAdjoint = false, spin_op op = spin_op()) override {

    // If there are extra controls, we are in control region and need to append
    // these controls.
    for (auto &e : extraControlIds)
      controls.emplace_back(2, e);

    // We need to check if we need take the adjoint of the operation. To do this
    // we use a logical XOR between `isAdjoint` and whether the size of
    // `adjointQueueStack` is even. The size of `adjointQueueStack` corresponds
    // to the number of nested `cudaq::adjoint` calls. If the size is even, then
    // we need to change the operation when `isAdjoint` is true. If the size is
    // odd, then we need to change the operation when `isAdjoint` is false.
    // (Adjoint modifiers cancel each other, e.g, `adj adj r1` is `r1`.)
    //
    // The cases:
    //  * not-adjoint, even number of `cudaq::adjoint` => _no_ need to change op
    //  * not-adjoint, odd number of `cudaq::adjoint`  => change op
    //  * adjoint,     even number of `cudaq::adjoint` => change op
    //  * adjoint,     odd number `cudaq::adjoint`     => _no_ need to change op
    //
    bool evenAdjointStack = (adjointQueueStack.size() % 2) == 0;
    if (isAdjoint != !evenAdjointStack) {
      for (std::size_t i = 0; i < params.size(); i++)
        params[i] = -1.0 * params[i];
      if (gateName == "t" || gateName == "s")
        gateName.append("dg");
    }

    if (!adjointQueueStack.empty()) {
      // Add to the adjoint instruction queue
      adjointQueueStack.back().emplace_back(gateName, params, controls, targets,
                                            op);
      return;
    }

    executeInstruction({std::move(gateName), params, controls, targets, op});
  }
};

