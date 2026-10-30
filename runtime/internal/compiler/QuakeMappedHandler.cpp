/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "llvm/ADT/StringExtras.h"

namespace {

/// The built-in handler from `quake-unmapped` to `quake-mapped`. It borrows
/// every qubit from a wire set, routes the kernel onto the profile's device,
/// and converts the swaps that routing adds to the basis.
class QuakeMappedHandler : public cudaq_internal::compiler::LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &input,
                const cudaq::config::Profile &output) const override {
    return std::holds_alternative<cudaq::config::QuakeUnmappedProfile>(input) &&
           std::holds_alternative<cudaq::config::QuakeMappedProfile>(output);
  }

  std::optional<cudaq_internal::compiler::StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    if (!supports(input, output))
      return std::nullopt;
    const auto &profile = std::get<cudaq::config::QuakeMappedProfile>(output);
    const auto level = static_cast<unsigned>(options.optimizationLevel);
    // An input with value semantics is already on wires, so this stage only
    // adds a wire set and assigns wire indices. `prepare-for-wireset` gives
    // the same IR on such input (`prepare_for_wireset_on_wires.qke`) but takes
    // longer than the rest of the stage on a large kernel. Any other input may
    // hold references, so the stage converts it to wires, unrolling the loops
    // that the input profile selects.
    const auto &unmapped = std::get<cudaq::config::QuakeUnmappedProfile>(input);
    std::vector<std::string> passes = {
        unmapped.semantics == cudaq::config::Semantics::value
            ? "add-wireset,func.func(assign-wire-indices)"
            : cudaq_internal::compiler::convertToWires(unmapped,
                                                       /*addWireSet=*/true)};
    // `raise-fatal-errors` makes a kernel that cannot be routed a compile
    // error instead of leaving it unrouted.
    // TODO: Search for a better layout with `placement=auto search=sabre` once
    // downstream code no longer assumes that mapped device qubits are
    // contiguous from 0 (NVIDIA/cuda-quantum#4753). A layout that is not the
    // identity also needs full-QIR codegen to allocate qubits by borrow
    // identity rather than in borrow order.
    if (profile.device)
      passes.push_back("qubit-mapping{device=" + *profile.device +
                       " raise-fatal-errors=true}");
    // The level-1 cleanup that every cost model shares.
    if (level >= 1)
      passes.push_back("two-qubit-gate-count-optimization-pipeline{level=1}");
    if (!profile.basisGates.empty())
      passes.push_back("basis-conversion{basis=" +
                       llvm::join(profile.basisGates, ",") + "}");
    cudaq_internal::compiler::StagePasses stagePasses;
    stagePasses.postDeployPasses = llvm::join(passes, ",");
    return stagePasses;
  }

  /// Borrowing every qubit needs each register's size, which often depends on
  /// an argument, whether or not the stage routes.
  bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const override {
    return supports(input, output);
  }
};

} // namespace

std::unique_ptr<cudaq_internal::compiler::LoweringHandler>
cudaq_internal::compiler::makeQuakeMappedHandler() {
  return std::make_unique<QuakeMappedHandler>();
}
