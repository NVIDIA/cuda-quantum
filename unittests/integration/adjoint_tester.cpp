/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/platform.h>

// Demonstrate we can perform multi-controlled operations
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
    cudaq::qvector q(5);

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

    x<cudaq::ctrl>(q[0], q[1]);
    cnot(q[0], q[1]);

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
CUDAQ_TEST(AdjointTester, checkSimple) {
  auto counts = cudaq::sample(1000, single_adjoint_test{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "0");

  auto counts2 = cudaq::sample(qvector_adjoint_test{});
  counts2.dump();
  EXPECT_EQ(1, counts2.size());
  EXPECT_TRUE(counts2.begin()->first == "00000");

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

CUDAQ_TEST(AdjointTester, checkNestedAdjoint) {

  struct xxxh_gates {
    void operator()(cudaq::qview<> &q) __qpu__ {
      x(q[2]);
      x(q[0], q[1]);
      h(q[2]);
    }
  };

  struct S_0 {
    void operator()(cudaq::qview<> q) __qpu__ {

      cudaq::compute_action([&]() { xxxh_gates{}(q); },
                            [&]() { x(q[0], q[1], q[2]); });
    }
  };

  struct P {
    void operator()(cudaq::qview<> q) __qpu__ { h(q[0], q[1]); }
  };

  struct R {
    void operator()(cudaq::qview<> q) __qpu__ {
      ry(M_PI / 16.0, q[2]);
      ry<cudaq::ctrl>(M_PI / 8.0, q[0], q[2]);
      ry<cudaq::ctrl>(M_PI / 4.0, q[1], q[2]);
    }
  };

  struct A {
    void operator()(cudaq::qview<> q) __qpu__ {

      P{}(q);
      R{}(q);
    }
  };

  struct S_chi {
    void operator()(cudaq::qview<> q) __qpu__ { z(q[2]); }
  };

  struct run_circuit {

    auto operator()(const int n_qubits, const int n_itrs) __qpu__ {
      cudaq::qvector q(n_qubits);

      A{}(q);

      for (int i = 0; i < n_itrs; ++i) {
        S_chi{}(q);
        cudaq::compute_action([&]() { cudaq::adjoint(A{}, q); },
                              [&]() { S_0{}(q); });
      }
      mz(q);
    }
  };

  auto counts = cudaq::sample(run_circuit{}, 3, 1);
  counts.dump();

  // Circuit should be
  // Just run this test with NVQPP_QIR_VERBOSE=1
  // We'll get this properly tested when we can
  // extract a better rep of the underlying circuit
  // { A
  // ctrl h 0 1
  // ry pi / 16 2
  // ctrl ry pi / 8 0 2
  // ctrl ry pi / 4 1 2
  // }
  // {S_chi
  //  z 2
  // }
  // {Compute, adjoint of A
  //  ctrl ry -pi/16 2
  //  ctrl ry -pi/8 0 2
  //  ctrl ry -pi/4 1 2
  //  ctrl h 0 1
  // }
  // {Action, S_0
  //  {Compute
  //    x 2
  //    x 0 1
  //    h 2
  //  }
  //  {Action
  //   x 0 1 2
  //  }
  //  { Uncompute
  //   h 2
  //   x 0 1
  //   x 2
  //   }
  // { Uncompute Adj A
  // ctrl h 0 1
  // ry pi / 16 2
  // ctrl ry pi / 8 0 2
  // ctrl ry pi / 4 1 2
  // }
}

#ifndef CUDAQ_BACKEND_DM

// From issue: https://github.com/NVIDIA/cuda-quantum/issues/1215

#ifdef CUDAQ_BACKEND_CUSTATEVEC_FP32
#define EPSILON std::numeric_limits<float>::epsilon()
#else
#define EPSILON std::numeric_limits<double>::epsilon()
#endif

// This implementation is based on:
// Knuth, D. E. (2014). Art of computer programming, volume 2: Seminumerical
//                      algorithms. Addison-Wesley Professional.
// (See page 233 for the discussion on floating points that was adapted here to
// implement the comparison for std::complex<double>. Here we use the Euclidean
// distance to compare the complex numbers.)
static bool essentially_equal(std::complex<double> a, std::complex<double> b,
                              double epsilon = EPSILON) {
  double tmp = std::min(std::abs(b), std::abs(a));
  return std::abs(a - b) <= (tmp * epsilon);
}

#undef EPSILON

static __qpu__ void foo(cudaq::qubit &q) { rz<cudaq::adj>(M_PI_2, q); }

static __qpu__ void bar() {
  cudaq::qubit q;
  rz<cudaq::adj>(M_PI_2, q);
  cudaq::adjoint(foo, q);
}

CUDAQ_TEST(AdjointTester, checkEvenAdjointNesting) {
  auto result = cudaq::get_state(bar);
  std::array<std::complex<double>, 2> expected = {1., 0};
  EXPECT_TRUE(essentially_equal(expected[0], result[0]));
  EXPECT_TRUE(essentially_equal(expected[1], result[1]));
}

static __qpu__ void zaz(cudaq::qubit &q) { rz<cudaq::adj>(M_PI_2, q); }

static __qpu__ void foo_2(cudaq::qubit &q) { cudaq::adjoint(zaz, q); }

static __qpu__ void bar_2() {
  cudaq::qubit q;
  rz(M_PI_2, q);
  cudaq::adjoint(foo_2, q);
}

CUDAQ_TEST(AdjointTester, checkOddAdjointNesting) {
  auto result = cudaq::get_state(bar_2);
  std::array<std::complex<double>, 2> expected = {1., 0};
  EXPECT_TRUE(essentially_equal(expected[0], result[0]));
  EXPECT_TRUE(essentially_equal(expected[1], result[1]));
}

#endif