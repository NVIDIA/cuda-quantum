/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/simulators.h"
#include "nvqir/AnalysisScope.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/resourcecounter/ResourceCounter.h"
#include "nvqir/resourcecounter/ResourceCounterScope.h"
#include <cudaq/algorithms/resource_estimation.h>
#include <stdexcept>
#include <utility>

namespace {
bool alwaysFalse() { return false; }
} // namespace

CUDAQ_TEST(AnalysisScopeTester, isActiveDuringLifetime) {
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
  {
    auto s = nvqir::resource_counter::make_scope(alwaysFalse);
    EXPECT_TRUE(nvqir::AnalysisScope::is_active());
    EXPECT_EQ(s.name(), "resource_counter");
  }
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, nestedThrows) {
  auto outer = nvqir::resource_counter::make_scope(alwaysFalse);
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());

  EXPECT_ANY_THROW(nvqir::resource_counter::make_scope(alwaysFalse));

  // Outer scope is still the active one after the failed nest attempt.
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, failedNestKeepsChoice) {
  bool outerCalled = false;
  bool innerCalled = false;

  auto outer = nvqir::resource_counter::make_scope([&outerCalled]() {
    outerCalled = true;
    return false;
  });

  EXPECT_ANY_THROW(nvqir::resource_counter::make_scope([&innerCalled]() {
    innerCalled = true;
    return true;
  }));
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());

  auto &sim = outer.simulator();
  const auto qIdx = sim.allocateQubit();
  EXPECT_FALSE(sim.mz(qIdx));
  EXPECT_TRUE(outerCalled);
  EXPECT_FALSE(innerCalled);
}

CUDAQ_TEST(AnalysisScopeTester, exitClearsCounts) {
  {
    auto s = nvqir::resource_counter::make_scope(alwaysFalse);
    cudaq::Resources counts;
    counts.appendInstruction("h", 0);
    nvqir::resource_counter::prepopulate(std::move(counts));
    EXPECT_EQ(nvqir::resource_counter::get_counts(s).count("h"), 1u);
  }
  auto next = nvqir::resource_counter::make_scope([] { return false; });
  EXPECT_EQ(nvqir::resource_counter::get_counts(next).count(), 0u);
}

CUDAQ_TEST(AnalysisScopeTester, prepopulateRejectsForeignScope) {
  auto *backendSim = cudaq::get_simulator();
  ASSERT_NE(backendSim, nvqir::getResourceCounterSimulator());

  nvqir::AnalysisScope s{"backend_scope", *backendSim, {}};
  cudaq::Resources counts;
  counts.appendInstruction("h", 0);
  EXPECT_ANY_THROW(nvqir::resource_counter::prepopulate(std::move(counts)));
}

CUDAQ_TEST(AnalysisScopeTester, getCountsRejectsForeignScope) {
  auto *backendSim = cudaq::get_simulator();
  ASSERT_NE(backendSim, nvqir::getResourceCounterSimulator());

  nvqir::AnalysisScope s{"backend_scope", *backendSim, {}};
  EXPECT_ANY_THROW(nvqir::resource_counter::get_counts(s));
}

CUDAQ_TEST(AnalysisScopeTester, recoversAfterThrowingChoice) {
  auto kernelWithMeasure = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };
  auto plainKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
  };

  // The choice function throws when invoked, which propagates out of
  // estimate_resources. The RAII scope must release on the way out.
  EXPECT_THROW(cudaq::estimate_resources(
                   []() -> bool { throw std::runtime_error("choice failed"); },
                   kernelWithMeasure),
               std::runtime_error);
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());

  // A subsequent estimate_resources on the same thread must work.
  auto resources = cudaq::estimate_resources(plainKernel);
  EXPECT_EQ(resources.count("h"), 1u);
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, releasesOnException) {
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
  try {
    auto s = nvqir::resource_counter::make_scope(alwaysFalse);
    EXPECT_TRUE(nvqir::AnalysisScope::is_active());
    throw std::runtime_error("simulated failure inside scope");
  } catch (const std::runtime_error &) {
    // expected
  }
  // RAII should have released the slot even though the body threw.
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, prepopulateNoScopeThrows) {
  EXPECT_FALSE(nvqir::AnalysisScope::is_active());
  cudaq::Resources counts;
  counts.appendInstruction("h", 0);
  EXPECT_ANY_THROW(nvqir::resource_counter::prepopulate(std::move(counts)));
}

CUDAQ_TEST(AnalysisScopeTester, prepopulateReflectsInCounts) {
  auto s = nvqir::resource_counter::make_scope(alwaysFalse);
  cudaq::Resources counts;
  counts.appendInstruction("h", 0);
  counts.appendInstruction("x", 1);
  nvqir::resource_counter::prepopulate(std::move(counts));

  auto observed = nvqir::resource_counter::get_counts(s);
  EXPECT_EQ(observed.count("h"), 1u);
  EXPECT_EQ(observed.count_controls("x", 1), 1u);
}

#ifdef CUDAQ_BACKEND_STIM
CUDAQ_TEST(AnalysisScopeTester, fromPluginStim) {
  // The Stim variant of the test executable links `nvqir-stim`, exporting
  // `getCircuitSimulator_stim` into the process. This is the dlsym path
  // that the upcoming DEM engine will rely on, so we exercise it now to
  // pin the design ahead of the DEM migration.
  auto s = nvqir::AnalysisScope::from_plugin("plugin_smoke", "stim");
  EXPECT_TRUE(nvqir::AnalysisScope::is_active());
  EXPECT_EQ(s.name(), "plugin_smoke");
  EXPECT_EQ(s.simulator().name(), "stim");
}
#endif
