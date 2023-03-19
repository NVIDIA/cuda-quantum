/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Test lambdas that are created within kernels and passed to both predefined
// kernels and user-defined kernels.

#include <cudaq.h>

struct test0 {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    auto lz = [](cudaq::qubit &q) __qpu__ { x(q); };
    cudaq::control(lz, q[0], q[1]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test0
// CHECK:           %[[VAL_2:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<?>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           %[[VAL_12:.*]] = quake.qextract %[[VAL_2]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           %[[VAL_15:.*]] = quake.qextract %[[VAL_2]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           quake.apply @__nvqpp__mlirgen__{{.*}}test0{{.*}}[%[[VAL_12]] : !quake.qref] %[[VAL_15]] : (!quake.qref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:        5test0
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.qref)
// CHECK:           quake.x (%[[VAL_0]])
// CHECK:           return
// CHECK:         }

struct test1 {
  void operator()() __qpu__ {
    cudaq::qreg<2> q;
    cudaq::control([](cudaq::qubit &q) __qpu__ { x(q); }, q[0], q[1]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1
// CHECK-SAME:        () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_3:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<2>
// CHECK:           %[[VAL_4:.*]] = cc.create_lambda {
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           %[[VAL_13:.*]] = quake.qextract %[[VAL_3]][%{{.*}}] : !quake.qvec<2>[i64] -> !quake.qref
// CHECK:           %[[VAL_16:.*]] = quake.qextract %[[VAL_3]][%{{.*}}] : !quake.qvec<2>[i64] -> !quake.qref
// CHECK:           quake.apply @__nvqpp__mlirgen__{{.*}}5test1{{.*}}[%[[VAL_13]] : !quake.qref] %[[VAL_16]] : (!quake.qref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:        5test1
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.qref)
// CHECK:           quake.x (%[[VAL_0]])
// CHECK:           return
// CHECK:         }

struct test2a {
  template <typename C>
  void operator()(C &&callme, cudaq::qreg<> &q) __qpu__ {
    callme(q[0]);
    callme(q[1]);
  }
};

struct test2b {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    test2a{}(
        [](cudaq::qubit &q) __qpu__ {
          h(q);
          y(q);
        },
        q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2b
// CHECK:           %[[VAL_2:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<?>
// CHECK:           %[[VAL_4:.*]] = cc.create_lambda {
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test2a{{.*}}(%{{.*}}, %[[VAL_2]]) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qvec<?>) -> ()
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test2a
// CHECK-SAME:       (%[[VAL_0:.*]]: !cc.lambda<(!quake.qref) -> ()>,
// CHECK-SAME:        %[[VAL_1:.*]]: !quake.qvec<?>)
// CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_1]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           call @__nvqpp__mlirgen__{{.*}}test2b{{.*}}(%[[VAL_4]]) : (!quake.qref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_1]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           call @__nvqpp__mlirgen__{{.*}}test2b{{.*}}(%[[VAL_7]]) : (!quake.qref) -> ()
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:        6test2b
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.qref)
// CHECK:           quake.h (%[[VAL_0]])
// CHECK:           quake.y (%[[VAL_0]])
// CHECK:           return
// CHECK:         }

struct test2a_c {
  template <typename C>
  void operator()(C &&callme, cudaq::qreg<> &q) __qpu__ {
    // void operator()(std::function<void(cudaq::qubit &)> &&callme,
    //            cudaq::qreg<> &q) __qpu__ {
    callme(q[0]);
    callme(q[1]);
  }
};

struct test2c {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    auto lz = [](cudaq::qubit &q) __qpu__ {
      h(q);
      z(q);
      h(q);
    };
    test2a_c{}(lz, q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2c
// CHECK:           %[[VAL_2:.*]] = quake.alloca(%{{.*}} : i64) : !quake.qvec<?>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test2a_c{{.*}}(%{{.*}}, %[[VAL_2]]) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qvec<?>) -> ()
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test2a_c
// CHECK-SAME:       (%[[VAL_0:.*]]: !cc.lambda<(!quake.qref) -> ()>,
// CHECK-SAME:        %[[VAL_1:.*]]: !quake.qvec<?>)
// CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_1]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           call @__nvqpp__mlirgen__{{.*}}test2c{{.*}}(%[[VAL_4]]) : (!quake.qref) -> ()
// CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_1]][%{{.*}}] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           call @__nvqpp__mlirgen__{{.*}}test2c{{.*}}(%[[VAL_7]]) : (!quake.qref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:        6test2c
// CHECK-SAME:        %[[VAL_0:.*]]: !quake.qref)
// CHECK:           quake.h (%[[VAL_0]])
// CHECK:           quake.z (%[[VAL_0]])
// CHECK:           quake.h (%[[VAL_0]])
// CHECK:           return
// CHECK:         }

struct test3a {
  void operator()(std::function<void(cudaq::qubit &)> &&callme,
		  cudaq::qreg<> &q) __qpu__ {
    callme(q[0]);
    callme(q[1]);
  }
};

struct test3 {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    auto lz = [](cudaq::qubit &q) __qpu__ {
      h(q);
      z(q);
      h(q);
    };
    test3a{}(lz, q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3a
// CHECK-SAME:       (%[[VAL_0:.*]]: !cc.lambda<(!quake.qref) -> ()>,
// CHECK-SAME:        %[[VAL_1:.*]]: !quake.qvec<?>) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_1]]{{\[}}%[[VAL_3]]] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_4]] : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_6:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_1]]{{\[}}%[[VAL_6]]] : !quake.qvec<?>[i64] -> !quake.qref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_7]] : (!cc.lambda<(!quake.qref) -> ()>, !quake.qref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3
// CHECK-SAME:        () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_1:.*]] = arith.extsi %[[VAL_0]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca(%[[VAL_1]] : i64) : !quake.qvec<?>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.qref):
// CHECK:             cc.scope {
// CHECK:               quake.h (%[[VAL_4]])
// CHECK:               quake.z (%[[VAL_4]])
// CHECK:               quake.h (%[[VAL_4]])
// CHECK:             }
// CHECK:           } : !cc.lambda<(!quake.qref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__test3a{{.*}}(%[[VAL_6:.*]], %[[VAL_2]]) : (!cc.lambda<(!quake.qref) -> ()>, !quake.qvec<?>) -> ()
// CHECK:           return
// CHECK:         }

struct test4x2 {
  template <typename C>
  void operator()(C &&callme, cudaq::qreg<> &q) __qpu__ {
    callme(q[0]);
    callme(q[1]);
  }
};

struct test4x4 {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    test4x2{}(
        [](cudaq::qubit &q) __qpu__ {
          h(q);
          y(q);
        },
        q);
  }
};

struct test4x8 {
  void operator()() __qpu__ {
    cudaq::qreg q(2);
    auto lz = [](cudaq::qubit &q) __qpu__ {
      h(q);
      z(q);
      h(q);
    };
    test4x2{}(lz, q);  // this is not the same instance as test4x4
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4x4
// CHECK:           call @__nvqpp__mlirgen__instance_test4x2{{.*}}test4x2{{.*}}test4
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test4x2
// CHECK-SAME:    7test4x2{{.*}}7test4x4
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x4
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x4
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:    7test4x4
// CHECK:           quake.h (%
// CHECK:           quake.y (%
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4x8
// CHECK:           call @__nvqpp__mlirgen__instance_test4x2{{.*}}7test4x2{{.*}}7test4x8
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test4x2
// CHECK-SAME:    7test4x2{{.*}}7test4x8
// CHECK:           call @__nvqpp__mlirgen__{{.*}}7test4x8
// CHECK:           call @__nvqpp__mlirgen__{{.*}}7test4x8
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__
// CHECK-SAME:    7test4x8
// CHECK:           quake.h (%
// CHECK:           quake.z (%
// CHECK:           quake.h (%
// CHECK:           return
// CHECK:         }

