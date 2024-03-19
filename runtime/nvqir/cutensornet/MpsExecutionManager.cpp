/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/ADT/StringSwitch.h"

namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}
namespace {
class MpsExecutionManager : public cudaq::BasicExecutionManager {
private:
  nvqir::CircuitSimulator *simulator() {
    return nvqir::getCircuitSimulatorInternal();
  }

  /// @brief To improve `qudit` allocation, we defer
  /// single `qudit` allocation requests until the first
  /// encountered `apply` call.
  std::vector<cudaq::QuditInfo> requestedAllocations;
  std::vector<cudaq::QuditInfo> auxQuditIdsForDeletion;

  std::vector<Instruction>
  decomposeMultiControlledInstruction(const Instruction &instruction,
                                      std::vector<cudaq::QuditInfo> &aux) {
    // Get the data, create the Qubit* targets
    auto [gateName, parameters, controls, targets, op] = instruction;
    if (controls.size() + targets.size() <= 2) {
      return {instruction};
    }
    std::vector<Instruction> decomposedInsts;

    const auto makeInstruction =
        [](const std::string gateName, const std::vector<double> &gateParams,
           const std::vector<cudaq::QuditInfo> &ctrls,
           const std::vector<cudaq::QuditInfo> &targets) -> Instruction {
      return std::make_tuple(gateName, gateParams, ctrls, targets, cudaq::spin_op{});
    };

    if (targets.size() > 1) {
      if (gateName == "swap") {
        {
          auto mutableCtrls = controls;
          mutableCtrls.emplace_back(targets[0]);
          const auto insts = decomposeMultiControlledInstruction(
              makeInstruction("x", {}, mutableCtrls, {targets[1]}),
              auxQuditIdsForDeletion);
          decomposedInsts.insert(decomposedInsts.end(), insts.begin(),
                                 insts.end());
        }
        {
          auto mutableCtrls = controls;
          mutableCtrls.emplace_back(targets[1]);
          const auto insts = decomposeMultiControlledInstruction(
              makeInstruction("x", {}, mutableCtrls, {targets[0]}),
              auxQuditIdsForDeletion);
          decomposedInsts.insert(decomposedInsts.end(), insts.begin(),
                                 insts.end());
        }
        {
          auto mutableCtrls = controls;
          mutableCtrls.emplace_back(targets[0]);
          const auto insts = decomposeMultiControlledInstruction(
              makeInstruction("x", {}, mutableCtrls, {targets[1]}),
              auxQuditIdsForDeletion);
          decomposedInsts.insert(decomposedInsts.end(), insts.begin(),
                                 insts.end());
        }
        return decomposedInsts;
      } else if (gateName == "exp_pauli") {
        if (controls.size() <= 1) {
          return {instruction};
        } else {
          std::vector<cudaq::QuditInfo> qubitSupport;
          std::vector<std::function<void(bool)>> basisChange;
          op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
            if (type != cudaq::pauli::I)
              qubitSupport.push_back(targets[qubitIdx]);

            if (type == cudaq::pauli::Y)
              basisChange.emplace_back([&, qubitIdx](bool reverse) {
                decomposedInsts.emplace_back(
                    makeInstruction("rx", {!reverse ? M_PI_2 : -M_PI_2}, {},
                                    {targets[qubitIdx]}));
              });
            else if (type == cudaq::pauli::X)
              basisChange.emplace_back([&, qubitIdx](bool) {
                decomposedInsts.emplace_back(
                    makeInstruction("h", {}, {}, {targets[qubitIdx]}));
              });
          });

          if (!basisChange.empty())
            for (auto &basis : basisChange)
              basis(false);

          std::vector<std::pair<cudaq::QuditInfo, cudaq::QuditInfo>> toReverse;
          for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
            decomposedInsts.emplace_back(makeInstruction(
                "x", {}, {qubitSupport[i]}, {qubitSupport[i + 1]}));
            toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
          }

          // Since this is a compute-action-uncompute type circuit, we only need
          // to apply control on this rz gate.
          {
            const auto mcRzInsts = decomposeMultiControlledInstruction(
                makeInstruction("rz", {-2.0 * parameters[0]}, controls,
                                {qubitSupport.back()}),
                auxQuditIdsForDeletion);
            decomposedInsts.insert(decomposedInsts.end(), mcRzInsts.begin(),
                                   mcRzInsts.end());
          }

          std::reverse(toReverse.begin(), toReverse.end());
          for (auto &[i, j] : toReverse)
            decomposedInsts.emplace_back(makeInstruction("x", {}, {i}, {j}));

          if (!basisChange.empty()) {
            std::reverse(basisChange.begin(), basisChange.end());
            for (auto &basis : basisChange)
              basis(true);
          }
          return decomposedInsts;
        }
      } else {
        throw std::runtime_error("Unsupported: " + gateName);
      }
    }

    const auto ccnot = [&](cudaq::QuditInfo &a, cudaq::QuditInfo &b,
                           cudaq::QuditInfo &c) {
      decomposedInsts.emplace_back(makeInstruction("h", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {b}, {c}));
      decomposedInsts.emplace_back(makeInstruction("tdg", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {a}, {c}));
      decomposedInsts.emplace_back(makeInstruction("t", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {b}, {c}));
      decomposedInsts.emplace_back(makeInstruction("tdg", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {a}, {c}));
      decomposedInsts.emplace_back(makeInstruction("t", {}, {}, {b}));
      decomposedInsts.emplace_back(makeInstruction("t", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("h", {}, {}, {c}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {a}, {b}));
      decomposedInsts.emplace_back(makeInstruction("t", {}, {}, {a}));
      decomposedInsts.emplace_back(makeInstruction("tdg", {}, {}, {b}));
      decomposedInsts.emplace_back(makeInstruction("x", {}, {a}, {b}));
    };

    const auto collectControls = [&](std::vector<cudaq::QuditInfo> &ctls,
                                     std::vector<cudaq::QuditInfo> &aux,
                                     int adjustment) {
      for (int i = 0; i < static_cast<int>(ctls.size()) - 1; i += 2) {
        ccnot(ctls[i], ctls[i + 1], aux[i / 2]);
      }
      for (int i = 0; i < static_cast<int>(ctls.size()) / 2 - 1 - adjustment;
           ++i) {
        ccnot(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
      }
    };
    const auto adjustForSingleControl =
        [&](std::vector<cudaq::QuditInfo> &ctls,
            std::vector<cudaq::QuditInfo> &aux) {
          if (ctls.size() % 2 != 0)
            ccnot(ctls[ctls.size() - 1], aux[ctls.size() - 3],
                  aux[ctls.size() - 2]);
        };
    for (std::size_t i = aux.size(); i < controls.size() - 1; ++i)
      aux.emplace_back(cudaq::QuditInfo(2, getAvailableIndex(2)));

    collectControls(controls, aux, 0);
    adjustForSingleControl(controls, aux);
    // Add to the instruction queue
    decomposedInsts.emplace_back(
        std::move(gateName), parameters,
        std::vector<cudaq::QuditInfo>{aux[controls.size() - 2]}, targets, op);
    adjustForSingleControl(controls, aux);
    collectControls(controls, aux, 0);
    return decomposedInsts;
  }

  /// @brief Allocate all requested `qudits`.
  void flushRequestedAllocations() {
    if (requestedAllocations.empty())
      return;

    allocateQudits(requestedAllocations);
    requestedAllocations.clear();
  }

protected:
  void allocateQudit(const cudaq::QuditInfo &q) override {
    requestedAllocations.emplace_back(2, q.id);
  }

  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    simulator()->allocateQubits(qudits.size());
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {

    // Before trying to deallocate, make sure the qudit hasn't
    // been requested but not allocated.
    auto iter =
        std::find(requestedAllocations.begin(), requestedAllocations.end(), q);
    if (iter != requestedAllocations.end()) {
      requestedAllocations.erase(iter);
      return;
    }

    simulator()->deallocate(q.id);
  }

  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    std::vector<std::size_t> local;
    for (auto &q : qudits) {
      auto iter = std::find(requestedAllocations.begin(),
                            requestedAllocations.end(), q);
      if (iter != requestedAllocations.end()) {
        requestedAllocations.erase(iter);
      } else {
        local.push_back(q.id);
      }
    }

    simulator()->deallocateQubits(local);
  }

  void handleExecutionContextChanged() override {
    requestedAllocations.clear();
    simulator()->setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    simulator()->resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

    // Get the data, create the Qubit* targets
    auto [gateName, parameters, controls, targets, op] = instruction;

    // Map the Qudits to Qubits
    std::vector<std::size_t> localT;
    std::transform(targets.begin(), targets.end(), std::back_inserter(localT),
                   [](auto &&el) { return el.id; });
    std::vector<std::size_t> localC;
    std::transform(controls.begin(), controls.end(), std::back_inserter(localC),
                   [](auto &&el) { return el.id; });

    // Apply the gate
    llvm::StringSwitch<std::function<void()>>(gateName)
        .Case("h", [&]() { simulator()->h(localC, localT[0]); })
        .Case("x", [&]() { simulator()->x(localC, localT[0]); })
        .Case("y", [&]() { simulator()->y(localC, localT[0]); })
        .Case("z", [&]() { simulator()->z(localC, localT[0]); })
        .Case("rx",
              [&]() { simulator()->rx(parameters[0], localC, localT[0]); })
        .Case("ry",
              [&]() { simulator()->ry(parameters[0], localC, localT[0]); })
        .Case("rz",
              [&]() { simulator()->rz(parameters[0], localC, localT[0]); })
        .Case("s", [&]() { simulator()->s(localC, localT[0]); })
        .Case("t", [&]() { simulator()->t(localC, localT[0]); })
        .Case("sdg", [&]() { simulator()->sdg(localC, localT[0]); })
        .Case("tdg", [&]() { simulator()->tdg(localC, localT[0]); })
        .Case("r1",
              [&]() { simulator()->r1(parameters[0], localC, localT[0]); })
        .Case("u1",
              [&]() { simulator()->u1(parameters[0], localC, localT[0]); })
        .Case("u3",
              [&]() {
                simulator()->u3(parameters[0], parameters[1], parameters[2],
                                localC, localT[0]);
              })
        .Case("swap",
              [&]() { simulator()->swap(localC, localT[0], localT[1]); })
        .Case("exp_pauli",
              [&]() {
                simulator()->applyExpPauli(parameters[0], localC, localT, op);
              })
        .Default([&]() {
          throw std::runtime_error("[DefaultExecutionManager] invalid gate "
                                   "application requested " +
                                   gateName + ".");
        })();
  }

  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    flushRequestedAllocations();
    return simulator()->mz(q.id, registerName);
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    flushRequestedAllocations();
    simulator()->flushGateQueue();

    if (executionContext->canHandleObserve) {
      auto result = simulator()->observe(*executionContext->spin.value());
      executionContext->expectationValue = result.expectationValue;
      executionContext->result = cudaq::sample_result(result);
      return;
    }

    assert(op.num_terms() == 1 && "Number of terms is not 1.");

    cudaq::info("Measure {}", op.to_string(false));
    std::vector<std::size_t> qubitsToMeasure;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitsToMeasure.push_back(qubitIdx);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          simulator()->rx(!reverse ? M_PI_2 : -M_PI_2, qubitIdx);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { simulator()->h(qubitIdx); });
    });

    // Change basis, flush the queue
    if (!basisChange.empty()) {
      for (auto &basis : basisChange)
        basis(false);

      simulator()->flushGateQueue();
    }

    // Get whether this is shots-based
    int shots = 0;
    if (executionContext->shots > 0)
      shots = executionContext->shots;

    // Sample and give the data to the context
    cudaq::ExecutionResult result = simulator()->sample(qubitsToMeasure, shots);
    executionContext->expectationValue = result.expectationValue;
    executionContext->result = cudaq::sample_result(result);

    // Restore the state.
    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);

      simulator()->flushGateQueue();
    }
  }

public:
  MpsExecutionManager() {
    cudaq::info("[MpsExecutionManager] Creating the {} backend.",
                simulator()->name());
  }
  virtual ~MpsExecutionManager() = default;

  void resetQudit(const cudaq::QuditInfo &q) override {
    flushRequestedAllocations();
    simulator()->resetQubit(q.id);
  }

  void endAdjointRegion() override {
    assert(!adjointQueueStack.empty() && "There must be at least one queue");

    auto adjointQueue = std::move(adjointQueueStack.back());
    adjointQueueStack.pop_back();

    // Select the queue to which these instructions will be added.
    InstructionQueue *queue = adjointQueueStack.empty()
                                  ? &instructionQueue
                                  : &(adjointQueueStack.back());

    std::reverse(adjointQueue.begin(), adjointQueue.end());
    for (auto &instruction : adjointQueue) {
      const auto insts = decomposeMultiControlledInstruction(
          instruction, auxQuditIdsForDeletion);
      queue->insert(queue->end(), insts.begin(), insts.end());
    }
  }

  /// The goal for apply is to create a new element of the
  /// instruction queue (a tuple).
  void apply(const std::string_view gateName, const std::vector<double> &params,
             const std::vector<cudaq::QuditInfo> &controls,
             const std::vector<cudaq::QuditInfo> &targets,
             bool isAdjoint, cudaq::spin_op op) override {

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
      for (std::size_t i = 0; i < params.size(); i++)
        mutable_params[i] = -1.0 * params[i];
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

    const auto insts = decomposeMultiControlledInstruction(
        {std::move(mutable_name), mutable_params, mutable_controls,
         mutable_targets, op},
        auxQuditIdsForDeletion);
    instructionQueue.insert(instructionQueue.end(), insts.begin(), insts.end());
  }

  void resetExecutionContext() override {
    BasicExecutionManager::resetExecutionContext();

    deallocateQudits(auxQuditIdsForDeletion);
    for (auto &q : auxQuditIdsForDeletion) {
      returnIndex(q.id);
    }
    auxQuditIdsForDeletion.clear();
  }
};

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(MpsExecutionManager)