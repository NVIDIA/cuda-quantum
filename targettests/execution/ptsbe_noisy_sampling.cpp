/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Integration tests for PTSBE noisy sampling with nvq++-compiled kernels.

#include <cudaq.h>
#include <cudaq/ptsbe/PTSBEExecutionData.h>
#include <cudaq/ptsbe/PTSBESample.h>
#include <cudaq/ptsbe/PTSBESampleResult.h>

#include <cstdio>

// ---------------------------------------------------------------------------
// GHZ with external depolarization noise
// ---------------------------------------------------------------------------
struct GHZKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  }
};

// ---------------------------------------------------------------------------
// Inline apply_noise with parameterized probabilities
// ---------------------------------------------------------------------------
struct InlineNoiseKernel {
  void operator()(double depol_prob, double bf_prob) __qpu__ {
    cudaq::qubit q;
    x(q);
    cudaq::apply_noise<cudaq::depolarization_channel>(depol_prob, q);
    cudaq::apply_noise<cudaq::bit_flip_channel>(bf_prob, q);
    mz(q);
  }
};

// ---------------------------------------------------------------------------
// Parameterized rotation with external gate and measurement noise
// ---------------------------------------------------------------------------
struct RotationKernel {
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    rx(theta, q[0]);
    ry(theta * 2.0, q[1]);
    // Measure each qubit on it's own rather than register
    mz(q[0]);
    mz(q[1]);
  }
};

// ---------------------------------------------------------------------------
// Implicit measurement (no mz call)
// ---------------------------------------------------------------------------
struct ImplicitMzKernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
bool testGHZWithNoise() {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("cx", cudaq::depolarization2(0.01));

  cudaq::ptsbe::sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto result = cudaq::ptsbe::sample(options, GHZKernel{});

  if (result.get_total_shots() != 100) {
    printf("GHZ FAIL: expected 100 shots, got %lu\n",
           (unsigned long)result.get_total_shots());
    return false;
  }

  auto count00000 = result.count("00000");
  auto count11111 = result.count("11111");
  if (count00000 + count11111 < 80) {
    printf("GHZ FAIL: dominant bitstrings too low (%lu + %lu = %lu)\n",
           (unsigned long)count00000, (unsigned long)count11111,
           (unsigned long)(count00000 + count11111));
    return false;
  }

  if (!result.has_execution_data()) {
    printf("GHZ FAIL: no execution data\n");
    return false;
  }

  const auto &data = result.execution_data();

  auto gateCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Gate);
  if (gateCount != 5) {
    printf("GHZ FAIL: expected 5 gates, got %lu\n", (unsigned long)gateCount);
    return false;
  }

  auto measCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Measurement);
  if (measCount != 5) {
    printf("GHZ FAIL: expected 5 measurements, got %lu\n",
           (unsigned long)measCount);
    return false;
  }

  return true;
}

bool testInlineNoise() {
  cudaq::ptsbe::sample_options options;
  options.shots = 100;

  options.ptsbe.return_execution_data = true;

  auto result = cudaq::ptsbe::sample(options, InlineNoiseKernel{}, 0.1, 0.05);

  if (result.get_total_shots() != 100) {
    printf("INLINE FAIL: expected 100 shots, got %lu\n",
           (unsigned long)result.get_total_shots());
    return false;
  }

  if (result.size() == 0) {
    printf("INLINE FAIL: no unique bitstrings\n");
    return false;
  }

  if (!result.has_execution_data()) {
    printf("INLINE FAIL: no execution data\n");
    return false;
  }

  const auto &data = result.execution_data();

  auto gateCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Gate);
  if (gateCount != 1) {
    printf("INLINE FAIL: expected 1 gate, got %lu\n",
           (unsigned long)gateCount);
    return false;
  }

  auto noiseCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Noise);
  if (noiseCount < 2) {
    printf("INLINE FAIL: expected at least 2 noise instructions, got %lu\n",
           (unsigned long)noiseCount);
    return false;
  }

  return true;
}

bool testParameterizedKernel() {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("rx", cudaq::depolarization_channel(0.02));
  noise.add_all_qubit_channel("ry", cudaq::depolarization_channel(0.02));
  noise.add_all_qubit_channel("mz", cudaq::bit_flip_channel(0.05));

  cudaq::ptsbe::sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto result = cudaq::ptsbe::sample(options, RotationKernel{}, 1.57);

  if (result.get_total_shots() != 100) {
    printf("PARAM FAIL: expected 100 shots, got %lu\n",
           (unsigned long)result.get_total_shots());
    return false;
  }

  if (!result.has_execution_data()) {
    printf("PARAM FAIL: no execution data\n");
    return false;
  }

  const auto &data = result.execution_data();

  auto gateCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Gate);
  if (gateCount != 2) {
    printf("PARAM FAIL: expected 2 gates, got %lu\n",
           (unsigned long)gateCount);
    return false;
  }

  return true;
}

bool testImplicitMeasurement() {
  cudaq::noise_model noise;
  // bit_flip(1.0) on mz is deterministic: flips every measurement outcome.
  // x|0> = |1>, then 100% bit-flip -> "0". No statistical variance.
  noise.add_channel("mz", {0}, cudaq::bit_flip_channel(1.0));

  cudaq::ptsbe::sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto result = cudaq::ptsbe::sample(options, ImplicitMzKernel{});

  if (result.get_total_shots() != 100) {
    printf("IMPLICIT_MZ FAIL: expected 100 shots, got %lu\n",
           (unsigned long)result.get_total_shots());
    return false;
  }

  auto count0 = result.count("0");
  if (count0 != 100) {
    printf("IMPLICIT_MZ FAIL: expected 100x '0', got %lu\n",
           (unsigned long)count0);
    return false;
  }

  if (!result.has_execution_data()) {
    printf("IMPLICIT_MZ FAIL: no execution data\n");
    return false;
  }

  const auto &data = result.execution_data();
  auto gateCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Gate);
  if (gateCount != 1) {
    printf("IMPLICIT_MZ FAIL: expected 1 gate, got %lu\n",
           (unsigned long)gateCount);
    return false;
  }

  auto measCount =
      data.count_instructions(cudaq::ptsbe::TraceInstructionType::Measurement);
  if (measCount != 1) {
    printf("IMPLICIT_MZ FAIL: expected 1 measurement, got %lu\n",
           (unsigned long)measCount);
    return false;
  }

  return true;
}

bool testAsyncSampling() {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("cx", cudaq::depolarization2(0.01));

  cudaq::ptsbe::sample_options options;
  options.shots = 200;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto future = cudaq::ptsbe::sample_async(options, GHZKernel{});
  auto result = future.get();

  if (result.get_total_shots() != 200) {
    printf("ASYNC FAIL: expected 200 shots, got %lu\n",
           (unsigned long)result.get_total_shots());
    return false;
  }

  if (!result.has_execution_data()) {
    printf("ASYNC FAIL: no execution data\n");
    return false;
  }

  if (result.size() == 0) {
    printf("ASYNC FAIL: no results\n");
    return false;
  }

  return true;
}

int main() {
  printf("STARTING GHZ\n");
  if (testGHZWithNoise())
    printf("GHZ_PASS\n");

  printf("STARTING INLINE\n");
  if (testInlineNoise())
    printf("INLINE_PASS\n");

  printf("STARTING PARAM\n");
  if (testParameterizedKernel())
    printf("PARAM_PASS\n");

  printf("STARTING IMPLICIT_MZ\n");
  if (testImplicitMeasurement())
    printf("IMPLICIT_MZ_PASS\n");

  printf("STARTING ASYNC\n");
  if (testAsyncSampling())
    printf("ASYNC_PASS\n");

  return 0;
}

// CHECK: GHZ_PASS
// CHECK: INLINE_PASS
// CHECK: PARAM_PASS
// CHECK: IMPLICIT_MZ_PASS
// CHECK: ASYNC_PASS
