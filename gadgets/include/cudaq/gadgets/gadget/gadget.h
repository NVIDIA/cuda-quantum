/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/gadgets/circuit/instructions.h"

namespace cudaq::gadgets {
inline namespace v0 {

struct Gadget;

/// XOR of distinct outcomes and a constant offset.
struct ParityExpr {
  std::vector<OutcomeRef> refs;
  bool offset = false;
};

/// Logical outcomes as parities of measurements m and input virtual outcomes
/// v_in: o = A m + B v_in + b.
using OutcomeMap = std::vector<std::pair<OutcomeRef, ParityExpr>>;

/// Signed product of generators by index: (-1)^negative * prod g_i.
struct GeneratorProduct {
  std::vector<std::size_t> generators;
  bool negative = false;
};

// TODO: validate(): sizes, commutation, decompositions.
struct SubsystemCode {
  std::string name;
  std::size_t n_phys = 0;
  std::size_t n_log = 0;
  std::size_t n_gauge = 0;
  std::vector<PauliString> stabilizers;
  std::vector<PauliString> logical_x;
  std::vector<PauliString> logical_z;
  std::vector<PauliString> gauge_x;
  std::vector<PauliString> gauge_z;
  std::vector<PauliString> anti_stabilizers;
  std::vector<PauliString> virtual_outcomes;
  /// One per virtual-outcome generator, in the independent stabilizer basis.
  std::vector<GeneratorProduct> decompositions;
  /// Products of virtual-outcome generators equal to (-1)^negative * I.
  std::vector<GeneratorProduct> metachecks;
  const DebugInfo *debug = nullptr;
};

// TODO: validate() against the gadget.
struct GadgetProfile {
  const Gadget *gadget = nullptr;
  std::vector<ParityExpr> detectors;
  std::vector<ParityExpr> success_bits;
  const DebugInfo *debug = nullptr;
};

/// Binds a code block, by index, to realization and logical-action qubits.
struct BlockBinding {
  std::size_t block = 0;
  bool output = false;
  std::vector<WireId> realization;
  std::vector<WireId> logical;
};

// TODO: validate(): code sizes, bindings, outcome-map shape, parameters.
struct Gadget {
  std::string name;
  std::vector<const SubsystemCode *> in_codes;
  std::vector<const SubsystemCode *> out_codes;
  SemanticCircuit logical_action;
  std::variant<SemanticCircuit, InstructionCircuit> realization;
  std::vector<BlockBinding> boundary_bindings;
  OutcomeMap outcome_map;
  std::vector<Param> params;
  const GadgetProfile *profile = nullptr;
  std::string schedule; // External artifact.
  const DebugInfo *debug = nullptr;

  const std::string &get_name() const { return name; }
};

/// Gadgets, by name.
using GadgetSet = InstructionSet<Gadget>;

} // namespace v0
} // namespace cudaq::gadgets
