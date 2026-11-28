/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/gadget/gadget.h"

#include <stdexcept>
#include <utility>

namespace cudaq::gadgets {
inline namespace v0 {

Gadget::Gadget(std::string name, std::vector<const SubsystemCode *> in_codes,
               std::vector<const SubsystemCode *> out_codes,
               SemanticCircuit logical_action, InstructionCircuit realization,
               OutcomeMap outcome_map)
    : name(std::move(name)), in_codes(std::move(in_codes)),
      out_codes(std::move(out_codes)),
      logical_action(std::move(logical_action)),
      realization(std::move(realization)), outcome_map(std::move(outcome_map)) {
  auto physical = [](const std::vector<const SubsystemCode *> &codes) {
    std::size_t total = 0;
    for (const SubsystemCode *code : codes)
      total += code->get_num_qubits();
    return total;
  };
  auto logical = [](const std::vector<const SubsystemCode *> &codes) {
    std::size_t total = 0;
    for (const SubsystemCode *code : codes)
      total += code->get_logicals().size();
    return total;
  };
  auto check = [this](std::size_t wires, std::size_t qubits, const char *what) {
    if (wires != qubits)
      throw std::invalid_argument("gadget '" + this->name + "': " + what +
                                  " do not match the codes");
  };
  const Wires &physical_wires = this->realization.get_wires();
  const Wires &logical_wires = this->logical_action.get_wires();
  check(physical_wires.q_in.size(), physical(this->in_codes),
        "realization inputs");
  check(physical_wires.q_out.size(), physical(this->out_codes),
        "realization outputs");
  check(logical_wires.q_in.size(), logical(this->in_codes),
        "logical action inputs");
  check(logical_wires.q_out.size(), logical(this->out_codes),
        "logical action outputs");
}

} // namespace v0
} // namespace cudaq::gadgets
