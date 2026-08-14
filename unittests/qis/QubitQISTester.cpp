/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

// Host-side quantum types (qubit.id(), qvector outside __qpu__, parameterized
// custom ops using std::exp on complex values) require the CUDAQ_LIBRARY_MODE
// inline QIS path. These tests stay here under the parent unittests/ g++
// build; the nvq++ kernel coverage lives in unittests/nvqpp/qis/.

CUDAQ_TEST(QubitQISTester, checkAllocateDeallocateSubRegister) {

  {
    cudaq::qubit q, r;
    EXPECT_EQ(q.id(), 0);
    EXPECT_EQ(r.id(), 1);

    h(q, r);
    cudaq::qvector qq(3);
    auto f = qq.front(2);
    h(f, qq[2]);
  }

  EXPECT_FALSE(cudaq::getExecutionManager()->memoryLeaked());

  {
    cudaq::qvector q(10);
    for (auto [i, q] : cudaq::enumerate(q)) {
      EXPECT_EQ(i, q.id());
    }

    cudaq::qubit r, s;
    EXPECT_EQ(r.id(), 10);
    EXPECT_EQ(s.id(), 11);

    // out of scope, qubits returned
  }
  EXPECT_FALSE(cudaq::getExecutionManager()->memoryLeaked());

  {
    cudaq::qvector q(15);
    EXPECT_EQ(q[14].id(), 14);

    EXPECT_EQ(q.front().id(), 0);
    EXPECT_EQ(q.back().id(), 14);
    auto f5 = q.front(5);
    EXPECT_EQ(f5.size(), 5);
    for (auto [i, qq] : cudaq::enumerate(f5)) {
      EXPECT_EQ(i, qq.id());
    }

    auto b4 = q.back(4);
    EXPECT_EQ(b4.size(), 4);
    EXPECT_EQ(b4[0].id(), 11);
    EXPECT_EQ(b4[1].id(), 12);
    EXPECT_EQ(b4[2].id(), 13);
    EXPECT_EQ(b4[3].id(), 14);
    EXPECT_EQ(b4.front().id(), 11);
    EXPECT_EQ(b4.back().id(), 14);

    auto view_from_span = b4.front(2);
    EXPECT_EQ(view_from_span[0].id(), 11);
    EXPECT_EQ(view_from_span[1].id(), 12);

    auto slice = q.slice(4, 7);
    EXPECT_EQ(slice.size(), 7);
    for (auto [i, qq] : cudaq::enumerate(slice)) {
      EXPECT_EQ(i + 4, qq.id());
    }

    auto slice_from_span = b4.slice(1, 2);
    EXPECT_EQ(slice_from_span.size(), 2);
    EXPECT_EQ(slice_from_span[0].id(), 12);
    EXPECT_EQ(slice_from_span[1].id(), 13);
  }

  EXPECT_FALSE(cudaq::getExecutionManager()->memoryLeaked());
}

CUDAQ_TEST(QubitQISTester, checkArray) {
  {
    cudaq::qarray<5> compileTimeQubits;
    EXPECT_EQ(compileTimeQubits.size(), 5);
    for (int i = 0; i < 5; i++)
      EXPECT_EQ(compileTimeQubits[i].id(), i);
  }

  {
    cudaq::qarray<15> q;
    EXPECT_EQ(q[14].id(), 14);

    EXPECT_EQ(q.front().id(), 0);
    EXPECT_EQ(q.back().id(), 14);
    auto f5 = q.front(5);
    EXPECT_EQ(f5.size(), 5);
    for (auto [i, qq] : cudaq::enumerate(f5)) {
      EXPECT_EQ(i, qq.id());
    }

    auto b4 = q.back(4);
    EXPECT_EQ(b4.size(), 4);
    EXPECT_EQ(b4[0].id(), 11);
    EXPECT_EQ(b4[1].id(), 12);
    EXPECT_EQ(b4[2].id(), 13);
    EXPECT_EQ(b4[3].id(), 14);
    EXPECT_EQ(b4.front().id(), 11);
    EXPECT_EQ(b4.back().id(), 14);

    auto view_from_span = b4.front(2);
    EXPECT_EQ(view_from_span[0].id(), 11);
    EXPECT_EQ(view_from_span[1].id(), 12);

    auto slice = q.slice(4, 7);
    EXPECT_EQ(slice.size(), 7);
    for (auto [i, qq] : cudaq::enumerate(slice)) {
      EXPECT_EQ(i + 4, qq.id());
    }
  }
}

using namespace std::complex_literals;

// Parameterized CustomU3 uses std::exp on complex values in the unitary
// generator; nvq++ cannot lower that. Keep this registration and its tests
// in library mode only.
CUDAQ_REGISTER_OPERATION(
    CustomU3, 1, 3,
    {std::cos(parameters[0] / 2.),
     -std::exp(1i * parameters[2]) * std::sin(parameters[0] / 2.),
     std::exp(1i * parameters[1]) * std::sin(parameters[0] / 2.),
     std::exp(1i * (parameters[2] + parameters[1])) *
         std::cos(parameters[0] / 2.)})

CUDAQ_TEST(CustomUnitaryTester, checkParameterized) {
  {
    // parameterized op, custom u3
    auto check_x = []() {
      cudaq::qubit q;
      // mimic Pauli-X gate
      CustomU3(M_PI, M_PI, M_PI_2, q);
    };
    auto counts = cudaq::sample(check_x);
    counts.dump();
    for (auto &[bits, count] : counts) {
      EXPECT_TRUE(bits == "1");
    }

    auto bell_pair = []() {
      cudaq::qvector qubits(2);
      // mimic Hadamard gate
      CustomU3(M_PI_2, 0., M_PI, qubits[0]);
      x<cudaq::ctrl>(qubits[0], qubits[1]);
    };
    counts = cudaq::sample(bell_pair);
    counts.dump();
    for (auto &[bits, count] : counts) {
      EXPECT_TRUE(bits == "00" || bits == "11");
    }

    // Can control
    auto another_bell_pair = []() {
      cudaq::qvector qubits(2);
      CustomU3(M_PI_2, 0., M_PI, qubits[0]);
      CustomU3<cudaq::ctrl>(M_PI, M_PI, M_PI_2, qubits[0], qubits[1]);
    };
    counts = cudaq::sample(another_bell_pair);
    counts.dump();
    for (auto &[bits, count] : counts) {
      EXPECT_TRUE(bits == "00" || bits == "11");
    }

    // can adjoint
    auto rotation_adjoint_test = []() {
      cudaq::qubit q;
      // mimic Rx gate
      CustomU3(1.1, -M_PI_2, M_PI_2, q);
      // rx<adj>(angle) = u3<adj>(angle, pi/2, -pi/2)
      CustomU3<cudaq::adj>(1.1, M_PI_2, -M_PI_2, q);
      // mimic Ry gate
      CustomU3(1.1, 0., 0., q);
      CustomU3<cudaq::adj>(1.1, 0., 0., q);
    };

    counts = cudaq::sample(rotation_adjoint_test);
    counts.dump();
    for (auto &[bits, count] : counts) {
      EXPECT_TRUE(bits == "0");
    }
  }
}
