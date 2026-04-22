/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test to verify MPSSimulationState::toHost correctly copies all data.
// This test reproduces a bug where memcpy used sizeof(ScalarType) instead of
// sizeof(std::complex<ScalarType>), resulting in only half the data being
// copied.

#include "CUDAQTestUtils.h"
#include <cmath>
#include <cudaq/algorithm.h>
#include <numeric>

#ifdef CUDAQ_BACKEND_TENSORNET_MPS

// Bell state kernel: creates (|00> + |11>) / sqrt(2)
__qpu__ void bell_state() {
  cudaq::qubit q, r;
  h(q);
  cx(q, r);
}

// GHZ state kernel for 4 qubits: creates (|0000> + |1111>) / sqrt(2)
__qpu__ void ghz_4() {
  cudaq::qvector q(4);
  h(q[0]);
  for (int i = 0; i < 3; i++)
    x<cudaq::ctrl>(q[i], q[i + 1]);
}

// Simple superposition: |+> state = (|0> + |1>) / sqrt(2)
__qpu__ void plus_state() {
  cudaq::qubit q;
  h(q);
}

CUDAQ_TEST(MPSToHostTester, checkToHostCopiesAllElements) {
  // Get state from Bell state circuit
  auto state = cudaq::get_state(bell_state);

  // Bell state should have 4 elements (2^2)
  const std::size_t numElements = 4;

  // Allocate buffer and initialize with sentinel values to detect incomplete
  // copy
  std::vector<std::complex<double>> hostBuffer(numElements, {-999.0, -999.0});

  // Check that state is on GPU (MPS states are device data)
  EXPECT_TRUE(state.is_on_gpu());

  // Copy state to host
  state.to_host(hostBuffer.data(), numElements);

  // Expected Bell state: (|00> + |11>) / sqrt(2)
  // State vector: [1/sqrt(2), 0, 0, 1/sqrt(2)]
  const double expected = 1.0 / std::sqrt(2.0);

  // Verify ALL elements, especially the last ones (index 2 and 3)
  // which would be corrupted if only half the bytes were copied
  EXPECT_NEAR(expected, hostBuffer[0].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[0].imag(), 1e-6);

  EXPECT_NEAR(0.0, hostBuffer[1].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[1].imag(), 1e-6);

  EXPECT_NEAR(0.0, hostBuffer[2].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[2].imag(), 1e-6);

  // This element would be wrong (sentinel value or garbage) if only half bytes
  // were copied
  EXPECT_NEAR(expected, hostBuffer[3].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[3].imag(), 1e-6);

  // Additional check: verify no sentinel values remain
  for (std::size_t i = 0; i < numElements; ++i) {
    EXPECT_NE(hostBuffer[i].real(), -999.0)
        << "Element " << i << " was not copied (sentinel value remains)";
  }
}

CUDAQ_TEST(MPSToHostTester, checkToHostLargerState) {
  // Test with a larger GHZ state (4 qubits = 16 elements)
  auto state = cudaq::get_state(ghz_4);

  const std::size_t numElements = 16; // 2^4

  // Initialize with sentinel values
  std::vector<std::complex<double>> hostBuffer(numElements, {-888.0, -888.0});

  EXPECT_TRUE(state.is_on_gpu());

  // Copy to host
  state.to_host(hostBuffer.data(), numElements);

  // GHZ state: (|0000> + |1111>) / sqrt(2)
  // Only first and last elements should be non-zero
  const double expected = 1.0 / std::sqrt(2.0);

  EXPECT_NEAR(expected, hostBuffer[0].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[0].imag(), 1e-6);

  // Middle elements should be zero
  for (std::size_t i = 1; i < numElements - 1; ++i) {
    EXPECT_NEAR(0.0, hostBuffer[i].real(), 1e-6)
        << "Element " << i << " should be 0";
    EXPECT_NEAR(0.0, hostBuffer[i].imag(), 1e-6);
  }

  // Last element - this would be corrupted with the bug
  EXPECT_NEAR(expected, hostBuffer[numElements - 1].real(), 1e-6)
      << "Last element should be 1/sqrt(2), but was corrupted";
  EXPECT_NEAR(0.0, hostBuffer[numElements - 1].imag(), 1e-6);

  // Verify no sentinel values remain (detects incomplete copy)
  for (std::size_t i = 0; i < numElements; ++i) {
    EXPECT_NE(hostBuffer[i].real(), -888.0)
        << "Element " << i << " was not copied";
  }
}

CUDAQ_TEST(MPSToHostTester, checkToHostSingleQubit) {
  // Test with single qubit |+> state
  auto state = cudaq::get_state(plus_state);

  const std::size_t numElements = 2; // 2^1

  std::vector<std::complex<double>> hostBuffer(numElements, {-777.0, -777.0});

  EXPECT_TRUE(state.is_on_gpu());

  state.to_host(hostBuffer.data(), numElements);

  // |+> state: (|0> + |1>) / sqrt(2)
  const double expected = 1.0 / std::sqrt(2.0);

  EXPECT_NEAR(expected, hostBuffer[0].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[0].imag(), 1e-6);

  // Second element would be wrong if only half was copied
  EXPECT_NEAR(expected, hostBuffer[1].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[1].imag(), 1e-6);
}

// Test for fp32 precision backend (tensornet-mps-fp32)
#ifdef CUDAQ_SIMULATION_SCALAR_FP32

__qpu__ void bell_fp32() {
  cudaq::qubit q, r;
  h(q);
  cx(q, r);
}

CUDAQ_TEST(MPSToHostTester, checkToHostFP32Precision) {
  // Test with fp32 precision - same bug affects fp32
  auto state = cudaq::get_state(bell_fp32);

  const std::size_t numElements = 4;

  // Use fp32 buffer with sentinel values
  std::vector<std::complex<float>> hostBuffer(numElements, {-999.0f, -999.0f});

  EXPECT_TRUE(state.is_on_gpu());

  state.to_host(hostBuffer.data(), numElements);

  // Bell state: [1/sqrt(2), 0, 0, 1/sqrt(2)]
  const float expected = 1.0f / std::sqrt(2.0f);

  EXPECT_NEAR(expected, hostBuffer[0].real(), 1e-5f);
  EXPECT_NEAR(0.0f, hostBuffer[1].real(), 1e-5f);
  EXPECT_NEAR(0.0f, hostBuffer[2].real(), 1e-5f);
  EXPECT_NEAR(expected, hostBuffer[3].real(), 1e-5f)
      << "FP32: Element 3 not copied correctly";

  // Verify no sentinel values remain
  for (std::size_t i = 0; i < numElements; ++i) {
    EXPECT_NE(hostBuffer[i].real(), -999.0f)
        << "FP32: Element " << i << " was not copied";
  }
}

#endif // CUDAQ_SIMULATION_SCALAR_FP32

#endif // CUDAQ_BACKEND_TENSORNET_MPS

// Test for tensornet (non-MPS) backend - same bug exists in
// tn_simulation_state.inc
#if defined(CUDAQ_BACKEND_TENSORNET) && !defined(CUDAQ_BACKEND_TENSORNET_MPS)

__qpu__ void bell_tensornet() {
  cudaq::qubit q, r;
  h(q);
  cx(q, r);
}

CUDAQ_TEST(TensorNetToHostTester, checkToHostCopiesAllElements) {
  // This test verifies the same bug fix for tensornet (non-MPS) backend
  auto state = cudaq::get_state(bell_tensornet);

  const std::size_t numElements = 4;
  std::vector<std::complex<double>> hostBuffer(numElements, {-999.0, -999.0});

  EXPECT_TRUE(state.is_on_gpu());

  state.to_host(hostBuffer.data(), numElements);

  const double expected = 1.0 / std::sqrt(2.0);

  EXPECT_NEAR(expected, hostBuffer[0].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[1].real(), 1e-6);
  EXPECT_NEAR(0.0, hostBuffer[2].real(), 1e-6);
  EXPECT_NEAR(expected, hostBuffer[3].real(), 1e-6)
      << "TensorNet: Element 3 not copied correctly";

  for (std::size_t i = 0; i < numElements; ++i) {
    EXPECT_NE(hostBuffer[i].real(), -999.0)
        << "TensorNet: Element " << i << " was not copied";
  }
}

#endif // CUDAQ_BACKEND_TENSORNET && !CUDAQ_BACKEND_TENSORNET_MPS
