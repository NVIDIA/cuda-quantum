/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "StimCircuitSimulator.cpp"
#include "common/ExecutionContext.h"
#include "nvqir/AnalysisScope.h"
#include "nvqir/QIRTypes.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

/// Wrapper to expose protected methods for testing.
class StimCircuitSimulatorTester : public nvqir::StimCircuitSimulator {
public:
  using GateApplicationTask =
      nvqir::CircuitSimulatorBase<double>::GateApplicationTask;

  void applyNamedGate(const std::string &name,
                      const std::vector<std::size_t> &controls,
                      const std::vector<std::size_t> &targets) {
    flushGateQueue();
    GateApplicationTask task(name, {}, controls, targets, {});
    applyGate(task);
  }

  void resetToZero() { setToZeroState(); }

  /// @brief Render `recordedCircuit` as a Stim-format string for inspection
  /// from the test body.
  std::string recordedCircuitText() const {
    std::stringstream ss;
    ss << recordedCircuit;
    return ss.str();
  }
};

// QEC declaration ops are wired into the Stim NVQIR backend by the runtime
// adapters in `runtime/nvqir/NVQIR.cpp`. Declared here so the test runner
// picks up the definitions from libnvqir; tests then exercise the full path
// from `extern "C"` symbol → `Result**`-to-`int64_t`-vector conversion →
// dispatch to the active simulator's QEC virtuals.
extern "C" void __quantum__qis__detector(Result **results, std::int64_t count);
extern "C" void __quantum__qis__logical_observable(Result **results,
                                                   std::int64_t count,
                                                   std::int64_t obs_index);
extern "C" void __quantum__qis__pair_detectors(Result **prev_results,
                                               std::int64_t prev_count,
                                               Result **curr_results,
                                               std::int64_t curr_count);

/// @brief Encode an `int64_t` chronological measurement index as the
/// `Result*` bit pattern the QIR ABI delivers to the QEC runtime adapter.
/// Mirrors the lowering pattern in `cudaq/test/Transforms/qir_api_qec.qke`.
static Result *measureIndexAsResultPtr(std::int64_t i) {
  return reinterpret_cast<Result *>(static_cast<std::intptr_t>(i));
}

CUDAQ_TEST(StimTester, TwoQubitPauliProducts) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();

  sim.applyNamedGate("IX", {}, {q0, q1});
  EXPECT_EQ(false, sim.mz(q0));
  EXPECT_EQ(true, sim.mz(q1));
}

CUDAQ_TEST(StimTester, IDGateMapping) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  sim.applyNamedGate("ID", {}, {q0});
  EXPECT_EQ(false, sim.mz(q0));
}

CUDAQ_TEST(StimTester, SetToZeroStatePreservesSimulators) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  for (int i = 0; i < 3; i++) {
    sim.x(q0);
    EXPECT_EQ(true, sim.mz(q0));
    sim.resetToZero();
  }
  EXPECT_EQ(false, sim.mz(q0));
}

CUDAQ_TEST(StimQECTester, DetectorEmitsRecordReferences) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);

  const std::int64_t indices[] = {0, 1};
  sim.detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos)
      << "recordedCircuit was:\n"
      << text;
}

CUDAQ_TEST(StimQECTester, LogicalObservableAccumulates) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);
  sim.mz(q2);

  const std::int64_t idsA[] = {0, 1};
  sim.logical_observable(idsA, 2, /*observable_index=*/0);
  const std::int64_t idsB[] = {2};
  sim.logical_observable(idsB, 1, /*observable_index=*/0);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("OBSERVABLE_INCLUDE(0) rec[-3] rec[-2]"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("OBSERVABLE_INCLUDE(0) rec[-1]"), std::string::npos)
      << text;
}

CUDAQ_TEST(StimQECTester, PairDetectorsEmitsOnePerPair) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  auto q3 = sim.allocateQubit();
  sim.mz(q0); // index 0
  sim.mz(q1); // index 1
  sim.mz(q2); // index 2
  sim.mz(q3); // index 3

  const std::int64_t prev[] = {0, 1};
  const std::int64_t curr[] = {2, 3};
  sim.pair_detectors(prev, curr, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-4] rec[-2]"), std::string::npos) << text;
  EXPECT_NE(text.find("DETECTOR rec[-3] rec[-1]"), std::string::npos) << text;
}

CUDAQ_TEST(StimQECTester, DetectorThrowsOnOutOfRangeIndex) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1); // num_measurements = 2

  const std::int64_t mixed[] = {0, 9};
  try {
    sim.detector(mixed, 2);
    FAIL() << "expected std::out_of_range";
  } catch (const std::out_of_range &e) {
    const std::string what(e.what());
    EXPECT_NE(what.find("detector"), std::string::npos) << what;
    EXPECT_NE(what.find("9"), std::string::npos) << what;
    EXPECT_NE(what.find("[0, 2)"), std::string::npos) << what;
  }

  EXPECT_EQ(sim.recordedCircuitText().find("DETECTOR"), std::string::npos)
      << sim.recordedCircuitText();
}

CUDAQ_TEST(StimQECTester, PairDetectorsAllOrNothingOnOutOfRange) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  auto q2 = sim.allocateQubit();
  auto q3 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);
  sim.mz(q2);
  sim.mz(q3); // num_measurements = 4

  const std::int64_t prev[] = {0, 1};
  const std::int64_t curr[] = {2, 99};
  EXPECT_ANY_THROW(sim.pair_detectors(prev, curr, 2));
  EXPECT_EQ(sim.recordedCircuitText().find("DETECTOR"), std::string::npos)
      << sim.recordedCircuitText();
}

CUDAQ_TEST(StimQECTester, PairDetectorsAdapterRejectsSizeMismatch) {
  Result *prev[1] = {measureIndexAsResultPtr(0)};
  Result *curr[2] = {measureIndexAsResultPtr(1), measureIndexAsResultPtr(2)};
  EXPECT_ANY_THROW(__quantum__qis__pair_detectors(prev, 1, curr, 2));
}

CUDAQ_TEST(StimQECTester, AdapterRejectsNegativeCount) {
  Result *results[1] = {measureIndexAsResultPtr(0)};
  EXPECT_ANY_THROW(__quantum__qis__detector(results, /*count=*/-1));
  EXPECT_ANY_THROW(__quantum__qis__logical_observable(results, /*count=*/-1,
                                                      /*obs_index=*/0));
}

CUDAQ_TEST(StimQECTester, AdapterRejectsNullBufferWithPositiveCount) {
  EXPECT_ANY_THROW(__quantum__qis__detector(nullptr, /*count=*/2));
}

CUDAQ_TEST(StimQECTester, AdapterRejectsNegativeObservableIndex) {
  Result *results[1] = {measureIndexAsResultPtr(0)};
  EXPECT_ANY_THROW(__quantum__qis__logical_observable(results, /*count=*/1,
                                                      /*obs_index=*/-1));
}

// End-to-end coverage of the NVQIR adapter
CUDAQ_TEST(StimQECTester, AdapterDispatchesToActiveSimulator) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  nvqir::AnalysisScope scope{"stim_qec_adapter_test", sim, {}};

  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0);
  sim.mz(q1);

  Result *indices[] = {measureIndexAsResultPtr(0), measureIndexAsResultPtr(1)};
  __quantum__qis__detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos) << text;
}

/// @brief RAII guard for the thread-local execution-context pointer used by
/// the simulator. Without this, an `ASSERT_*` failure inside the test would
/// abort before `resetExecutionContext`, leaving the global pointer dangling
/// to a destructed `ExecutionContext` and corrupting downstream tests.
class ScopedExecutionContext {
public:
  explicit ScopedExecutionContext(cudaq::ExecutionContext &ctx) {
    cudaq::detail::setExecutionContext(&ctx);
  }
  ~ScopedExecutionContext() { cudaq::detail::resetExecutionContext(); }
  ScopedExecutionContext(const ScopedExecutionContext &) = delete;
  ScopedExecutionContext &operator=(const ScopedExecutionContext &) = delete;
};

CUDAQ_TEST(StimQECTester, DetectorFlushesPendingSampleMeasurements) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);

  cudaq::ExecutionContext ctx("sample", /*shots=*/1);
  ctx.explicitMeasurements = true;
  sim.configureExecutionContext(ctx);
  ScopedExecutionContext ctxGuard{ctx};

  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();
  sim.mz(q0, "h0");
  sim.mz(q1, "h1");

  // Before the QEC call, the deferred `M` op is not yet in the recorded
  // circuit.
  EXPECT_EQ(sim.recordedCircuitText().find("M "), std::string::npos);

  const std::int64_t indices[] = {0, 1};
  sim.detector(indices, 2);

  const auto text = sim.recordedCircuitText();
  const auto mPos = text.find("M ");
  const auto dPos = text.find("DETECTOR");
  ASSERT_NE(mPos, std::string::npos) << text;
  ASSERT_NE(dPos, std::string::npos) << text;
  EXPECT_LT(mPos, dPos) << "M must land before DETECTOR:\n" << text;
  EXPECT_NE(text.find("DETECTOR rec[-2] rec[-1]"), std::string::npos) << text;
}
