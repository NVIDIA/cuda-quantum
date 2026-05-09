/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/analysis/resource_counter.h"
#include "cudaq/analysis/scope.h"
#include "nvqir/CircuitSimulator.h"
#include <stdexcept>
#include <string>

namespace {
bool alwaysFalse() { return false; }
} // namespace

CUDAQ_TEST(AnalysisScopeTester, scopeIsActiveDuringLifetime) {
  EXPECT_FALSE(cudaq::analysis::scope::is_active());
  {
    auto s = cudaq::analysis::resource_counter::make_scope(alwaysFalse);
    EXPECT_TRUE(cudaq::analysis::scope::is_active());
    EXPECT_EQ(s.name(), "resource_counter");
  }
  EXPECT_FALSE(cudaq::analysis::scope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, nestedScopeThrows) {
  auto outer = cudaq::analysis::resource_counter::make_scope(alwaysFalse);
  EXPECT_TRUE(cudaq::analysis::scope::is_active());

  EXPECT_THROW(
      cudaq::analysis::resource_counter::make_scope(alwaysFalse),
      std::runtime_error);

  // Outer scope is still the active one after the failed nest attempt.
  EXPECT_TRUE(cudaq::analysis::scope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, scopeReleasedOnExceptionInBody) {
  EXPECT_FALSE(cudaq::analysis::scope::is_active());
  try {
    auto s = cudaq::analysis::resource_counter::make_scope(alwaysFalse);
    EXPECT_TRUE(cudaq::analysis::scope::is_active());
    throw std::runtime_error("simulated failure inside scope");
  } catch (const std::runtime_error &) {
    // expected
  }
  // RAII should have released the slot even though the body threw.
  EXPECT_FALSE(cudaq::analysis::scope::is_active());
}

CUDAQ_TEST(AnalysisScopeTester, prepopulateWithoutScopeThrows) {
  EXPECT_FALSE(cudaq::analysis::scope::is_active());
  cudaq::Resources counts;
  counts.appendInstruction("h", 0);
  EXPECT_THROW(cudaq::analysis::resource_counter::prepopulate(std::move(counts)),
               std::runtime_error);
}

CUDAQ_TEST(AnalysisScopeTester, prepopulateInsideScopeReflectsInGetCounts) {
  auto s = cudaq::analysis::resource_counter::make_scope(alwaysFalse);
  cudaq::Resources counts;
  counts.appendInstruction("h", 0);
  counts.appendInstruction("x", 1);
  cudaq::analysis::resource_counter::prepopulate(std::move(counts));

  auto observed = cudaq::analysis::resource_counter::get_counts(s);
  EXPECT_EQ(observed.count("h"), 1u);
  EXPECT_EQ(observed.count_controls("x", 1), 1u);
}

#ifdef CUDAQ_BACKEND_STIM
CUDAQ_TEST(AnalysisScopeTester, fromPluginResolvesStim) {
  // The Stim variant of the test executable links `nvqir-stim`, exporting
  // `getCircuitSimulator_stim` into the process. This is the dlsym path
  // that the upcoming DEM engine will rely on, so we exercise it now to
  // pin the design ahead of the DEM migration.
  auto s = cudaq::analysis::scope::from_plugin("plugin_smoke", "stim");
  EXPECT_TRUE(cudaq::analysis::scope::is_active());
  EXPECT_EQ(s.name(), "plugin_smoke");
  EXPECT_EQ(s.simulator().name(), "stim");
}
#endif
