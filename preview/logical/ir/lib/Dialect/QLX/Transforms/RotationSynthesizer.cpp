// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// Licensed under the Apache License, Version 2.0.

#include "qlx/Dialect/QLX/Transforms/RotationSynthesizer.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include <cmath>
#include <map>
#include <mutex>

using namespace mlir;
using namespace qlx;

namespace {
class GridSynth final : public RotationSynthesizer {
public:
  llvm::StringRef name() const override { return "gridsynth"; }
  llvm::StringRef version() const override { return "builtin"; }
  llvm::StringRef lastError() const override {
    return "GridSynth failed to synthesize the rotation";
  }
  FailureOr<SynthesizedWord> synthesizeRz(double angle,
                                          double precision) override {
    cudaq::synth::Real epsilon(precision);
    auto bits = static_cast<mpfr_prec_t>(
        64 + std::ceil(-std::log2(precision)) +
        (angle == 0 ? 0 : std::max(0, std::ilogb(std::abs(angle)) + 1)));
    cudaq::synth::ScopedDefaultPrecision workingPrecision(
        std::max(bits, cudaq::synth::details::required_precision(epsilon)));
    cudaq::synth::GridsynthOptions options;
    options.seed = 0;
    auto circuit =
        cudaq::synth::gridsynth(cudaq::synth::Real(angle), epsilon, options);
    if (failed(circuit))
      return failure();
    SynthesizedWord word;
    word.algorithm = "ross-selinger-gridsynth";
    word.implementation = "cudaq-synth";
    word.version = "builtin";
    // GridSynth stores a matrix product, opposite to QLX's chronological order.
    for (auto it = circuit->rbegin(); it != circuit->rend(); ++it) {
      switch (*it) {
      case cudaq::synth::Gate::H:
        word.actions.push_back(BuiltinAction::h);
        break;
      case cudaq::synth::Gate::S:
        word.actions.push_back(BuiltinAction::s);
        break;
      case cudaq::synth::Gate::T:
        word.actions.push_back(BuiltinAction::t);
        break;
      case cudaq::synth::Gate::X:
        word.actions.append({BuiltinAction::h, BuiltinAction::s,
                             BuiltinAction::s, BuiltinAction::h});
        break;
      case cudaq::synth::Gate::W:
        break;
      }
    }
    return word;
  }
};

struct Registry {
  std::mutex mutex;
  std::map<std::string, std::shared_ptr<RotationSynthesizer>> entries;
  Registry() { entries.emplace("gridsynth", std::make_shared<GridSynth>()); }
};

Registry &registry() {
  // Python-backed entries are explicitly removed at interpreter exit. Avoid
  // static destruction ordering against Python and the shared MLIR runtime.
  static auto *value = new Registry;
  return *value;
}
} // namespace

bool qlx::registerRotationSynthesizer(
    std::shared_ptr<RotationSynthesizer> value) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  return r.entries.emplace(value->name().str(), std::move(value)).second;
}

std::shared_ptr<RotationSynthesizer>
qlx::lookupRotationSynthesizer(llvm::StringRef name) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  auto it = r.entries.find(name.str());
  return it == r.entries.end() ? nullptr : it->second;
}

void qlx::unregisterRotationSynthesizer(llvm::StringRef name) {
  if (name == "gridsynth")
    return;
  std::shared_ptr<RotationSynthesizer> removed;
  auto &r = registry();
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.entries.find(name.str());
    if (it != r.entries.end()) {
      removed = std::move(it->second);
      r.entries.erase(it);
    }
  }
  // Destroy callbacks outside the registry lock.
}
