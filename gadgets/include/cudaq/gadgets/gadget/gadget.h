/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/instructions.h"
#include "cudaq/gadgets/codes/subsystem_code.h"

namespace cudaq::gadgets {
inline namespace v0 {

/// XOR of distinct outcomes and a constant offset.
struct ParityExpr {
  std::vector<OutcomeRef> refs;
  bool offset = false;
};

/// Logical outcomes as parities of measurements m and input virtual outcomes
/// v_in: o = A m + B v_in + b.
using OutcomeMap = std::vector<std::pair<OutcomeRef, ParityExpr>>;

/// Code blocks bind to wires by position: the realization's qubit inputs are
/// the in_codes' physical qubits in order, and the logical action's are their
/// logical qubits; outputs likewise with out_codes.
// TODO: validate(): outcome-map shape.
struct Gadget {
  /// Throws std::invalid_argument unless the realization's qubit inputs
  /// (outputs) number the input (output) codes' physical qubits, and the
  /// logical action's number their logical qubits.
  Gadget(std::string name, std::vector<const SubsystemCode *> in_codes,
         std::vector<const SubsystemCode *> out_codes,
         SemanticCircuit logical_action, InstructionCircuit realization,
         OutcomeMap outcome_map = {});

  const std::string &get_name() const { return name; }
  const std::vector<const SubsystemCode *> &get_in_codes() const {
    return in_codes;
  }
  const std::vector<const SubsystemCode *> &get_out_codes() const {
    return out_codes;
  }
  const SemanticCircuit &get_logical_action() const { return logical_action; }
  const InstructionCircuit &get_realization() const { return realization; }
  const OutcomeMap &get_outcome_map() const { return outcome_map; }

private:
  std::string name;
  std::vector<const SubsystemCode *> in_codes;
  std::vector<const SubsystemCode *> out_codes;
  SemanticCircuit logical_action;
  InstructionCircuit realization;
  OutcomeMap outcome_map;
};

/// The redundancies in a gadget that let us detect and correct errors. Each
/// entry is an extended outcome parity: over the realization's outcomes and
/// the input codes' virtual outcomes.
// TODO: validate() against the target: outcome references exist, and one
// output syndrome per independent generator of the output codes.
struct GadgetContract {
  GadgetContract(const Gadget &target, std::vector<ParityExpr> detectors,
                 std::vector<ParityExpr> inactive_detectors = {},
                 std::vector<ParityExpr> output_syndromes = {})
      : target(&target), detectors(std::move(detectors)),
        inactive_detectors(std::move(inactive_detectors)),
        output_syndromes(std::move(output_syndromes)) {}

  const Gadget &get_target() const { return *target; }
  const std::vector<ParityExpr> &get_detectors() const { return detectors; }
  const std::vector<ParityExpr> &get_inactive_detectors() const {
    return inactive_detectors;
  }
  const std::vector<ParityExpr> &get_output_syndromes() const {
    return output_syndromes;
  }

private:
  const Gadget *target;
  std::vector<ParityExpr> detectors;
  std::vector<ParityExpr> inactive_detectors;
  std::vector<ParityExpr> output_syndromes;
};

} // namespace v0
} // namespace cudaq::gadgets
