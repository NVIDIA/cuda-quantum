/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <numeric>

#include <cudaq/algorithms/observe.h>
#include <cudaq/operators.h>

#ifndef CUDAQ_BACKEND_DM
// Host-side quantum type tests (allocation, qarray, parameterized CustomU3)
// live in unittests/qis/ . This file covers kernels compiled with nvq++.

CUDAQ_TEST(QubitQISTester, checkCommonKernel) {
  auto ghz = []() __qpu__ {
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

#ifndef CUDAQ_BACKEND_STIM
  auto ansatz = [](double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto energy = cudaq::observe(ansatz, h, .59);
  EXPECT_NEAR(energy, -1.7487, 1e-3);
#endif
}

#ifndef CUDAQ_BACKEND_STIM
namespace qubit_qis_tester {
// Local lambdas passed to cudaq::control / cudaq::adjoint are lifted to
// separate functions; apply-op-specialization then fails on those calls
// because inlining runs later in the nvq++ pipeline. Use named noinline
// __qpu__ helpers instead (same pattern as adjoint_tester / grover_test).
// See https://github.com/NVIDIA/cuda-quantum/issues/3762.
__attribute__((noinline)) __qpu__ void apply_x(cudaq::qubit &q) { x(q); }

__attribute__((noinline)) __qpu__ void test_inner_adjoint(cudaq::qubit &q) {
  cudaq::adjoint(apply_x, q);
}

__attribute__((noinline)) __qpu__ void ctrl_x(cudaq::qubit &ctrl,
                                              cudaq::qubit &target) {
  cudaq::control(apply_x, ctrl, target);
}

struct ccnot_test {
  void operator()() __qpu__ {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    auto controls = q.front(2);
    cudaq::control(test_inner_adjoint, controls, q[2]);

    mz(q);
  }
};

struct nested_ctrl {
  void operator()() __qpu__ {
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
    cudaq::control(ctrl_x, q[0], q[1], q[2]);

    mz(q);
  }
};
} // namespace qubit_qis_tester

CUDAQ_TEST(QubitQISTester, checkCtrlRegion) {

  auto ccnot = []() __qpu__ {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    x<cudaq::ctrl>(q[0], q[1], q[2]);

    mz(q);
  };

  auto counts = cudaq::sample(ccnot);
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "101");

  auto counts2 = cudaq::sample(qubit_qis_tester::ccnot_test{});
  EXPECT_EQ(1, counts2.size());
  EXPECT_TRUE(counts2.begin()->first == "101");

  auto counts3 = cudaq::sample(qubit_qis_tester::nested_ctrl{});
  EXPECT_EQ(1, counts3.size());
  EXPECT_TRUE(counts3.begin()->first == "101");
}
#endif

#ifndef CUDAQ_BACKEND_STIM
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
#endif

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

#ifndef CUDAQ_BACKEND_STIM
CUDAQ_TEST(QubitQISTester, checkU3Op) {
  auto check_x = []() __qpu__ {
    cudaq::qubit q;
    // mimic Pauli-X gate
    u3(M_PI, M_PI, M_PI_2, q);
  };
  auto counts = cudaq::sample(check_x);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "1");
  }

  auto bell_pair = []() __qpu__ {
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
#endif

#ifndef CUDAQ_BACKEND_STIM
CUDAQ_TEST(QubitQISTester, checkU3Ctrl) {
  auto another_bell_pair = []() __qpu__ {
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
#endif

#ifndef CUDAQ_BACKEND_STIM
CUDAQ_TEST(QubitQISTester, checkU3Adj) {
  auto rotation_adjoint_test = []() __qpu__ {
    cudaq::qubit q;
    // mimic Rx gate
    u3(1.1, -M_PI_2, M_PI_2, q);
    // rx<adj>(angle) = u3<adj>(angle, -pi/2, pi/2)
    u3<cudaq::adj>(1.1, -M_PI_2, M_PI_2, q);
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
#endif

#ifndef CUDAQ_BACKEND_STIM

// Test someone can build a library of custom operations
CUDAQ_REGISTER_OPERATION(
    /* Name */ CustomHadamard, /*NumTargets*/ 1, /*NumParameters*/ 0,
    /* Unitary Generator */ {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2});
CUDAQ_REGISTER_OPERATION(CustomX, 1, 0, {0, 1, 1, 0});
CUDAQ_REGISTER_OPERATION(CustomCNOT, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0});
// Parameterized CustomU3 (std::exp on complex values) lives in
// unittests/qis/ under CUDAQ_LIBRARY_MODE.
CUDAQ_REGISTER_OPERATION(CustomSwap, 2, 0,
                         {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1})

CUDAQ_TEST(CustomUnitaryTester, checkBasic) {
  {
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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

CUDAQ_TEST(CustomUnitaryTester, checkMultiQubitOps) {
  {
    // Test swap operation
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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
    auto kernel = []() __qpu__ {
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
#endif
