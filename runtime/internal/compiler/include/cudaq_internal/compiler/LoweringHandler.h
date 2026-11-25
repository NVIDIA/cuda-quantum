/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/Profile/AllProfiles.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
struct CompileOptions;
}

namespace cudaq_internal::compiler {

/// The passes one compilation stage adds to the JIT pipeline. The pipeline
/// runs `canonicalize`, prep, deploy, and finalize once for the whole target,
/// and each field says where a stage's passes go:
///
/// \code
///   canonicalize
///   prep pipeline
///     preDeployPasses       (first stage only)
///   jit-deploy-pipeline     (only when a stage needs gate lowering)
///     postDeployPasses
///   jit-finalize-pipeline
///     postFinalizePasses    (last stage only)
/// \endcode
///
/// The first-stage and last-stage limits keep each stage's passes after those
/// of the stages before it. Without them, a second stage's passes before
/// deploy would run ahead of the first stage's passes after deploy.
/// `getPassPipeline` checks each stage and rejects a target that breaks either
/// limit.
struct StagePasses {
  /// Passes that run after prep and before deploy.
  std::string preDeployPasses{};
  /// Passes that run after deploy and before finalize.
  std::string postDeployPasses{};
  /// Passes that run after finalize.
  std::string postFinalizePasses{};
  /// The stage's output is limited to a basis, so prep and deploy lower the
  /// gates that hardware lacks: custom unitaries, state initialization, `u3`,
  /// and multi-controlled gates.
  // TODO: Lower these gates in each profile's lowering, driven by what its
  // basis does not accept, once no custom stage relies on prep and deploy
  // for them. This field then goes.
  bool needsGateLowering = true;
  /// The stage normalizes the kernel: it converts the kernel to wires with
  /// `prepare-for-wireset` and applies the requested level. When every stage
  /// does, `codegen` only emits.
  bool normalizesKernel = false;
};

/// Interface for lowering between IR profiles.
///
/// Registering lowering handlers with the compiler is what enables the JIT
/// compiler to lower between different IR profiles. Whenever new optimization
/// pipelines or support for new IR profiles is added, a corresponding lowering
/// handler must be defined and registered.
class LoweringHandler {
public:
  virtual ~LoweringHandler() = default;

  /// Whether this handler lowers \p input into \p output. The compiler uses
  /// the first handler that says yes for a stage.
  ///
  /// This takes no compile options, so the same stage always gets the same
  /// handler. `specializesArguments` looks handlers up before a launch's
  /// options exist, and it must find the same handler that the compiler
  /// uses. A handler can still choose its passes from the options in
  /// `getStagePasses`.
  virtual bool supports(const cudaq::config::Profile &input,
                        const cudaq::config::Profile &output) const = 0;

  /// Return the passes this handler adds to lower \p input into \p output, or
  /// `nullopt` if this handler cannot perform that lowering.
  virtual std::optional<StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const = 0;

  /// Whether this stage needs the values of the kernel's arguments. If it
  /// does, the compiler replaces each argument with its value before the
  /// stage runs. For example, Clifford+T synthesis has to know every rotation
  /// angle, and an angle often comes from an argument, such as a QFT's width.
  ///
  /// Answer from the profiles alone, not the compile options. Python asks
  /// this same question to decide how to pack the arguments, and it does that
  /// before the launch's options exist. An answer that depended on the
  /// options could disagree with what the compiler does.
  virtual bool
  requiresArgumentValues(const cudaq::config::Profile &input,
                         const cudaq::config::Profile &output) const {
    return false;
  }
};

/// Whether a stage into \p profile compiles for a simulator. `gate-count` is
/// the cost model for simulators. `get_state` returns every qubit the kernel
/// allocates, so such a stage keeps the implicit output that holds them.
// TODO: This choice does not belong to the cost model. Erase the implicit
// output per launch instead, for the launches that do not read idle qubits.
inline bool
compilesForSimulator(const cudaq::config::QuakeUnmappedProfile &profile) {
  return profile.costModel == cudaq::config::CostModel::gate_count;
}

/// Pipeline text for `prepare-for-wireset`, which unrolls the loops that
/// \p profile selects, up to its limit, and moves the kernel's qubits onto
/// wires. A qubit picked by a longer loop or a run-time index stays a
/// reference. Stages run it at every optimization level, since the level only
/// changes how much we optimize, not the representation. With \p addWireSet,
/// it also borrows every qubit from a new wire set, which `qubit-mapping`
/// needs. With \p decomposeExpPauli, it decomposes each `exp_pauli` whose
/// Pauli word is known. Otherwise `exp_pauli` and its qubits stay on
/// references. When \p profile compiles for a simulator, the kernel keeps the
/// implicit output that holds every qubit, so that `dqe` removes no idle
/// qubit.
inline std::string
convertToWires(const cudaq::config::QuakeUnmappedProfile &profile,
               bool addWireSet, bool decomposeExpPauli) {
  return std::string("prepare-for-wireset{") +
         (addWireSet ? "add-wireset=true " : "") +
         "maximum-iterations=" + std::to_string(profile.maxUnrollIterations) +
         (profile.unrollLoops == cudaq::config::LoopUnrolling::index_use
              ? " unroll-only-index-use-loops=true"
              : "") +
         (decomposeExpPauli ? "" : " decompose-exp-pauli=false") +
         (compilesForSimulator(profile) ? " erase-implicit-output=false" : "") +
         "}";
}

/// Return the first handler in \p handlers supporting the conversion
/// \p input -> \p output, or nullptr if none does.
const LoweringHandler *selectLoweringHandler(
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    const cudaq::config::Profile &input, const cudaq::config::Profile &output);

/// The compiler's built-in lowering handlers, in selection order.
std::vector<std::unique_ptr<LoweringHandler>> makeBuiltinLoweringHandlers();

/// The built-in generic handler lowering to the quake-reference-semantics
/// profile.
std::unique_ptr<LoweringHandler> makeQuakeReferenceSemanticsHandler();

/// The built-in handler lowering to the `quake-unmapped` profile.
std::unique_ptr<LoweringHandler> makeQuakeUnmappedHandler();

/// The built-in handler lowering from `quake-unmapped` to `quake-mapped`.
std::unique_ptr<LoweringHandler> makeQuakeMappedHandler();

} // namespace cudaq_internal::compiler
