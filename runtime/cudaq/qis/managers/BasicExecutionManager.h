/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "cudaq/operators.h"
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
  /// @brief Return true if we are in tracer mode
  bool isInTracerMode() {
    return executionContext && executionContext->name == "tracer";
  }

  /// @brief An instruction is composed of a operation name,
  /// a optional set of rotation parameters, control qudits,
  /// target qudits, and an optional spin_op.
  using Instruction = std::tuple<std::string, std::vector<double>,
                                 std::vector<cudaq::QuditInfo>,
                                 std::vector<cudaq::QuditInfo>, spin_op_term>;

  /// @brief `typedef` for a queue of instructions
  using InstructionQueue = std::vector<Instruction>;

  /// @brief The current execution context, e.g. sampling or observation
  cudaq::ExecutionContext *executionContext = nullptr;

  /// @brief Store qudits for delayed deletion under certain execution contexts
  std::vector<QuditInfo> contextQuditIdsForDeletion;

  /// @brief The current queue of operations to execute
  InstructionQueue instructionQueue;

  /// @brief When adjoint operations are requested we can store them here for
  /// delayed execution
  std::vector<InstructionQueue> adjointQueueStack;

  /// @brief When we are in a control region, we need to store extra control
  /// qudit ids.
  std::vector<std::size_t> extraControlIds;

  /// @brief Subtype-specific qudit allocation method
  virtual void allocateQudit(const QuditInfo &q) = 0;

  /// @brief Allocate a set of `qudits` with a single call.
  virtual void allocateQudits(const std::vector<QuditInfo> &qudits) = 0;

  /// @brief Subtype specific qudit deallocation method
  virtual void deallocateQudit(const QuditInfo &q) = 0;

  /// @brief Subtype specific qudit deallocation, deallocate
  /// all qudits in the vector.
  virtual void deallocateQudits(const std::vector<QuditInfo> &qudits) = 0;

  /// @brief Subtype-specific handler for when the execution context changes
  virtual void handleExecutionContextChanged() = 0;

  /// @brief Subtype-specific handler for when the current execution context has
  /// ended.
  virtual void handleExecutionContextEnded() = 0;

  /// @brief Subtype-specific method for affecting the execution of a queued
  /// instruction.
  virtual void executeInstruction(const Instruction &inst) = 0;

  /// @brief Subtype-specific method for performing qudit measurement.
  virtual int measureQudit(const cudaq::QuditInfo &q,
                           const std::string &registerName) = 0;

  /// @brief Measure the state in the respective basis described each term in
  /// the given `spin_op`.
  virtual void measureSpinOp(const cudaq::spin_op &op) = 0;

  /// @brief Subtype-specific method for performing qudit reset.
  virtual void resetQudit(const QuditInfo &q) = 0;

public:
  BasicExecutionManager() = default;
  virtual ~BasicExecutionManager() = default;

  void setExecutionContext(cudaq::ExecutionContext *_ctx) override {
    executionContext = _ctx;
    handleExecutionContextChanged();
    instructionQueue.clear();
  }

  void resetExecutionContext() override {
    ScopedTraceWithContext("BasicExecutionManager::resetExecutionContext");
    synchronize();

    if (!executionContext)
      return;

    // Do any final post-processing before
    // we deallocate the qudits
    handleExecutionContextEnded();

    deallocateQudits(contextQuditIdsForDeletion);
    for (auto &q : contextQuditIdsForDeletion)
      returnIndex(q.id);

    contextQuditIdsForDeletion.clear();
    executionContext = nullptr;
  }

  std::size_t allocateQudit(std::size_t quditLevels) override {
    auto new_id = getNextIndex();
    if (isInTracerMode())
      return new_id;
    allocateQudit({quditLevels, new_id});
    return new_id;
  }

  void returnQudit(const QuditInfo &qid) override {
    if (!executionContext) {
      deallocateQudit(qid);
      returnIndex(qid.id);
      return;
    }

    if (isInTracerMode()) {
      returnIndex(qid.id);
      return;
    }

    contextQuditIdsForDeletion.push_back(qid);
  }

  void startAdjointRegion() override { adjointQueueStack.emplace_back(); }

  void endAdjointRegion() override {
    assert(!adjointQueueStack.empty() && "There must be at least one queue");

    auto adjointQueue = std::move(adjointQueueStack.back());
    adjointQueueStack.pop_back();

    // Select the queue to which these instructions will be added.
    InstructionQueue *queue = adjointQueueStack.empty()
                                  ? &instructionQueue
                                  : &(adjointQueueStack.back());

    std::reverse(adjointQueue.begin(), adjointQueue.end());
    for (auto &instruction : adjointQueue)
      queue->push_back(instruction);
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
  void apply(const std::string_view gateName, const std::vector<double> &params,
             const std::vector<cudaq::QuditInfo> &controls,
             const std::vector<cudaq::QuditInfo> &targets,
             bool isAdjoint = false,
             spin_op_term op = cudaq::spin_op::identity()) override {

    // Make a copy of the name that we can mutate if necessary
    std::string mutable_name(gateName);

    // Make a copy of the parameters that we can mutate
    std::vector<double> mutable_params = params;

    // Create an array of controls, we will
    // prepend any extra controls if in a control region
    std::vector<cudaq::QuditInfo> mutable_controls;
    for (auto &e : extraControlIds)
      mutable_controls.emplace_back(2, e);

    for (auto &e : controls)
      mutable_controls.push_back(e);

    std::vector<cudaq::QuditInfo> mutable_targets;
    for (auto &t : targets)
      mutable_targets.push_back(t);

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
      if (gateName == "u3") {
        mutable_params[0] = -1.0 * params[0];
        mutable_params[1] = -1.0 * params[2];
        mutable_params[2] = -1.0 * params[1];
      } else if (gateName == "u2") {
        mutable_params[0] = -1.0 * params[1] - M_PI;
        mutable_params[1] = -1.0 * params[0] + M_PI;
      } else {
        for (std::size_t i = 0; i < params.size(); i++)
          mutable_params[i] = -1.0 * params[i];
      }
      if (gateName == "t")
        mutable_name = "tdg";
      else if (gateName == "s")
        mutable_name = "sdg";
    }

    if (!adjointQueueStack.empty()) {
      // Add to the adjoint instruction queue
      adjointQueueStack.back().emplace_back(
          mutable_name, mutable_params, mutable_controls, mutable_targets, op);
      return;
    }

    // Add to the instruction queue
    instructionQueue.emplace_back(std::move(mutable_name), mutable_params,
                                  mutable_controls, mutable_targets, op);
  }

  void applyNoise(const kraus_channel &channel,
                  const std::vector<QuditInfo> &targets) override {
    return;
  }

  void synchronize() override {
    for (auto &instruction : instructionQueue) {
      if (!isInTracerMode()) {
        executeInstruction(instruction);
        continue;
      }

      auto &&[name, params, controls, targets, op] = instruction;
      executionContext->kernelTrace.appendInstruction(name, params, controls,
                                                      targets);
    }
    instructionQueue.clear();
  }

  int measure(const cudaq::QuditInfo &target,
              const std::string registerName = "") override {
    if (isInTracerMode())
      return 0;

    // We hit a measure, need to exec / clear instruction queue
    synchronize();

    // Instruction executed, run the measure call
    return measureQudit(target, registerName);
  }

  cudaq::SpinMeasureResult measure(const cudaq::spin_op &op) override {
    synchronize();
    measureSpinOp(op);
    return std::make_pair(executionContext->expectationValue.value(),
                          executionContext->result);
  }

  void reset(const QuditInfo &target) override {
    if (isInTracerMode())
      return;
    // We hit a reset, need to exec / clear instruction queue
    synchronize();
    resetQudit(target);
  }
};

} // namespace cudaq
