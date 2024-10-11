/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <numeric>

#include <cudaq/algorithms/observe.h>
#include <cudaq/spin_op.h>

#ifndef CUDAQ_BACKEND_DM
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

CUDAQ_TEST(QubitQISTester, checkCommonKernel) {
  auto ghz = []() {
    const int N = 5;
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  };
  auto counts = cudaq::sample(ghz);
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == "00000" || bits == "11111");
  }
  EXPECT_EQ(counter, 1000);

  auto ansatz = [](double theta) {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  auto energy = cudaq::observe(ansatz, h, .59);
  EXPECT_NEAR(energy, -1.7487, 1e-3);
}

CUDAQ_TEST(QubitQISTester, checkCtrlRegion) {

  auto ccnot = []() {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    x<cudaq::ctrl>(q[0], q[1], q[2]);

    mz(q);
  };

  struct ccnot_test {
    void operator()() __qpu__ {
      cudaq::qvector q(3);

      x(q);
      x(q[1]);

      auto apply_x = [](cudaq::qubit &q) { x(q); };
      auto test_inner_adjoint = [&](cudaq::qubit &q) {
        cudaq::adjoint(apply_x, q);
      };

      auto controls = q.front(2);
      cudaq::control(test_inner_adjoint, controls, q[2]);

      mz(q);
    }
  };

  struct nested_ctrl {
    void operator()() __qpu__ {
      auto apply_x = [](cudaq::qubit &r) { x(r); };

      cudaq::qvector q(3);
      // Create 101
      x(q);
      x(q[1]);

      // Fancy nested CCX
      // Walking inner nest to outer
      // 1. Queue X(q[2])
      // 2. Queue Ctrl (q[1]) X (q[2])
      // 3. Queue Ctrl (q[0], q[1]) X(q[2]);
      // 4. Apply
      cudaq::control(
          [&](cudaq::qubit &r) {
            cudaq::control([&](cudaq::qubit &r) { apply_x(r); }, q[1], r);
          },
          q[0], q[2]);

      mz(q);
    }
  };

  auto counts = cudaq::sample(ccnot);
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "101");

  auto counts2 = cudaq::sample(ccnot_test{});
  EXPECT_EQ(1, counts2.size());
  EXPECT_TRUE(counts2.begin()->first == "101");

  auto counts3 = cudaq::sample(nested_ctrl{});
  EXPECT_EQ(1, counts3.size());
  EXPECT_TRUE(counts3.begin()->first == "101");
}

CUDAQ_TEST(QubitQISTester, checkAdjointRegions) {
  struct single_adjoint_test {
    void operator()() __qpu__ {
      cudaq::qubit q;

      x(q);
      x<cudaq::adj>(q);

      mz(q);
    }
  };

  struct qvector_adjoint_test {
    void operator()() __qpu__ {
      cudaq::qvector q(10);

      x(q);
      x<cudaq::adj>(q);

      mz(q);
    }
  };

  struct rotation_adjoint_test {
    void operator()() __qpu__ {
      cudaq::qvector q(1);

      rx(1.1, q[0]);
      rx<cudaq::adj>(1.1, q[0]);

      ry(1.1, q[0]);
      ry<cudaq::adj>(1.1, q[0]);

      rz(1.1, q[0]);
      rz<cudaq::adj>(1.1, q[0]);
      mz(q);
    }
  };

  struct twoqbit_adjoint_test {
    void operator()() __qpu__ {
      cudaq::qvector q(2);

      cnot(q[0], q[1]);
      cnot(q[0], q[1]);

      cx(q[0], q[1]);
      cx(q[0], q[1]);

      cy(q[0], q[1]);
      cy(q[0], q[1]);

      cz(q[0], q[1]);
      cz(q[0], q[1]);

      x<cudaq::adj>(q[0], q[1]);
      cx(q[0], q[1]);

      mz(q);
    }
  };

  struct test_adjoint {
    void operator()(cudaq::qview<> q) __qpu__ {
      h(q[0]);
      t(q[1]);
      s(q[2]);
    }
  };

  struct test_cudaq_adjoint {
    void operator()() __qpu__ {
      cudaq::qvector q(3);
      x(q[0]);
      x(q[2]);
      test_adjoint{}(q);
      cudaq::adjoint(test_adjoint{}, q);
      mz(q);
    }
  };

  auto counts = cudaq::sample(single_adjoint_test{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "0");

  auto counts2 = cudaq::sample(qvector_adjoint_test{});
  counts2.dump();
  EXPECT_EQ(1, counts2.size());
  EXPECT_TRUE(counts2.begin()->first == "0000000000");

  auto counts3 = cudaq::sample(rotation_adjoint_test{});
  counts3.dump();
  EXPECT_EQ(1, counts3.size());
  EXPECT_TRUE(counts3.begin()->first == "0");

  auto counts4 = cudaq::sample(twoqbit_adjoint_test{});
  counts4.dump();
  EXPECT_EQ(1, counts4.size());
  EXPECT_TRUE(counts4.begin()->first == "00");

  auto counts5 = cudaq::sample(test_cudaq_adjoint{});
  counts5.dump();
  EXPECT_EQ(1, counts5.size());
  EXPECT_TRUE(counts5.begin()->first == "101");
}

CUDAQ_TEST(QubitQISTester, checkMeasureResetFence) {
  {
    struct init_measure {
      auto operator()() __qpu__ {
        // Allocate then measure, no gates.
        // Check that allocation requests are flushed.
        cudaq::qvector q(2);
        mz(q);
      }
    };
    auto kernel = init_measure{};
    auto counts = cudaq::sample(kernel);
    EXPECT_EQ(1, counts.size());
    EXPECT_TRUE(counts.begin()->first == "00");
  }
  {
    struct reset_circ {
      auto operator()() __qpu__ {
        cudaq::qvector q(2);
        x(q);
        reset(q[0]);
        mz(q);
      }
    };
    auto kernel = reset_circ{};
    auto counts = cudaq::sample(kernel);
    EXPECT_EQ(1, counts.size());
    // |11> -> |01> after reset
    EXPECT_TRUE(counts.begin()->first == "01");
  }
}

CUDAQ_TEST(QubitQISTester, checkU3Op) {
  auto check_x = []() {
    cudaq::qubit q;
    // mimic Pauli-X gate
    u3(M_PI, M_PI, M_PI_2, q);
  };
  auto counts = cudaq::sample(check_x);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "1");
  }

  auto bell_pair = []() {
    cudaq::qvector qubits(2);
    // mimic Hadamard gate
    u3(M_PI_2, 0., M_PI, qubits[0]);
    x<cudaq::ctrl>(qubits[0], qubits[1]);
  };
  counts = cudaq::sample(bell_pair);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
}

CUDAQ_TEST(QubitQISTester, checkU3Ctrl) {
  auto another_bell_pair = []() {
    cudaq::qvector qubits(2);
    u3(M_PI_2, 0., M_PI, qubits[0]);
    u3<cudaq::ctrl>(M_PI, M_PI, M_PI_2, qubits[0], qubits[1]);
  };
  auto counts = cudaq::sample(another_bell_pair);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
}

CUDAQ_TEST(QubitQISTester, checkU3Adj) {
  auto rotation_adjoint_test = []() {
    cudaq::qubit q;
    // mimic Rx gate
    u3(1.1, -M_PI_2, M_PI_2, q);
    // rx<adj>(angle) = u3<adj>(angle, pi/2, -pi/2)
    u3<cudaq::adj>(1.1, M_PI_2, -M_PI_2, q);
    // mimic Ry gate
    u3(1.1, 0., 0., q);
    u3<cudaq::adj>(1.1, 0., 0., q);
  };

  auto counts = cudaq::sample(rotation_adjoint_test);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "0");
  }
}

using namespace std::complex_literals;

// Test someone can build a library of custom operations
CUDAQ_REGISTER_OPERATION(
    /* Name */ CustomHadamard, /*NumTargets*/ 1, /*NumParameters*/ 0,
    /* Unitary Generator */ {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2});
CUDAQ_REGISTER_OPERATION(CustomX, 1, 0, {0, 1, 1, 0});
CUDAQ_REGISTER_OPERATION(CustomCNOT, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0});
CUDAQ_REGISTER_OPERATION(
    CustomU3, 1, 3,
    {std::cos(parameters[0] / 2.),
     -std::exp(1i * parameters[2]) * std::sin(parameters[0] / 2.),
     std::exp(1i * parameters[1]) * std::sin(parameters[0] / 2.),
     std::exp(1i * (parameters[2] + parameters[1])) *
         std::cos(parameters[0] / 2.)})
CUDAQ_REGISTER_OPERATION(CustomSwap, 2, 0,
                         {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1})

CUDAQ_TEST(CustomUnitaryTester, checkBasic) {
  {
    auto kernel = []() {
      cudaq::qubit q, r;
      CustomHadamard(q);
      CustomCNOT(q, r);
    };

    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "00" || k == "11");
    }
    EXPECT_EQ(counter, 1000);
  }
  {
    // Can be controlled
    auto kernel = []() {
      cudaq::qubit q, r;
      x(q);
      CustomX<cudaq::ctrl>(q, r);
    };

    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "11");
    }
    EXPECT_EQ(counter, 1000);
  }
  {
    // Can be controlled with negation
    auto kernel = []() {
      cudaq::qubit q, r;
      CustomX<cudaq::ctrl>(!q, r);
    };

    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "01");
    }
    EXPECT_EQ(counter, 1000);
  }
}

/// NOTE: This is supported only in library mode
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

CUDAQ_TEST(CustomUnitaryTester, checkMultiQubitOps) {
  {
    // Test swap operation
    auto kernel = []() {
      cudaq::qubit q, r;
      x(q);             // q -> 1, r -> 0
      CustomSwap(q, r); // q -> 0 , r -> 1
    };
    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "01");
    }
    EXPECT_EQ(counter, 1000);
  }
// NOTE: 'cutensornetStateApplyControlledTensorOperator' can only handle single
// target, hence, multi-qubit controlled custom operations not supported on
// tensornet backends
#ifndef CUDAQ_BACKEND_TENSORNET
  {
    // Multi-qubit can be controlled, with one-control
    auto kernel = []() {
      cudaq::qvector q(3);
      x(q[0]);
      x(q[1]);
      CustomCNOT<cudaq::ctrl>(q[0], q[1], q[2]);
    };
    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "111");
    }
    EXPECT_EQ(counter, 1000);
  }
  {
    // Multi-qubit can be controlled, with multi-qubit control
    auto kernel = []() {
      cudaq::qvector q(4);
      x(q.front(3));
      CustomCNOT<cudaq::ctrl>(q[0], q[1], q[2], q[3]);
    };
    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "1111");
    }
    EXPECT_EQ(counter, 1000);
  }
  {
    // Test controlled swap operation
    auto kernel = []() {
      cudaq::qubit q, r, c;
      x(q);
      CustomSwap<cudaq::ctrl>(c, q, r); // no swap
    };
    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "100");
    }
    EXPECT_EQ(counter, 1000);
  }
  {
    // Test multi-controlled swap operation
    auto kernel = []() {
      cudaq::qvector q(4);
      x(q.front(3));
      CustomSwap<cudaq::ctrl>(q[0], q[1], q[2], q[3]); // swap q[3] and q[2]
    };
    auto counts = cudaq::sample(kernel);
    counts.dump();
    int counter = 0;
    for (auto &[k, v] : counts) {
      counter += v;
      EXPECT_TRUE(k == "1101");
    }
    EXPECT_EQ(counter, 1000);
  }
#endif
}

#endif
