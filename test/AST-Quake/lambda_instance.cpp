/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test0()
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_2:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.apply @__nvqpp__mlirgen__ZN5test0[[LAM1:.*]] [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN5test0
// CHECK-SAME:      [[LAM1]](%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

struct test1 {
  void operator()() __qpu__ {
    cudaq::qreg<2> q;
    cudaq::control([](cudaq::qubit &q) __qpu__ { x(q); }, q[0], q[1]);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_2:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.apply @__nvqpp__mlirgen__ZN5test1[[LAM1:.*]] [%[[VAL_3]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN5test1
// CHECK-SAME:      [[LAM1]](%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2b() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"test2a" {}>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test2aZN6test2b[[LAM2A:.*]](%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN6test2b
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.y %[[VAL_0]] : (!quake.ref) -> ()
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2c() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.z %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.struct<"test2a_c" {}>
// CHECK:           call @__nvqpp__mlirgen__instance_test2a_cRZN6test2c[[LAM2C:.*]](%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN6test2c
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.z %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3a(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_2]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           cc.call_callable %[[VAL_0]], %[[VAL_3]] : (!cc.callable<(!quake.ref) -> ()>, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.z %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.struct<"test3a" {}>
// CHECK:           call @__nvqpp__mlirgen__test3a(%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN5test3cl
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.z %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

struct test4x2 {
  template <typename C>
  void operator()(C &&callme, cudaq::qreg<> &q) __qpu__ {
    callme(q[0]);
    callme(q[1]);
  }
};

// Template member is deferred. See below.

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4x4() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<"test4x2" {}>
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           call @__nvqpp__mlirgen__instance_test4x2Z[[LAM42a:.*]](%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN7test4x4
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.y %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4x8() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = cc.create_lambda {
// CHECK:           ^bb0(%[[VAL_3:.*]]: !quake.ref{{.*}}):
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.z %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:             quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           } : !cc.callable<(!quake.ref) -> ()>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.struct<"test4x2" {}>
// CHECK:           call @__nvqpp__mlirgen__instance_test4x2R[[LAM42b:.*]](%[[VAL_5:.*]], %[[VAL_1]]) : (!cc.callable<(!quake.ref) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ZN7test4x8
// CHECK-SAME:      _(%[[VAL_0:.*]]: !quake.ref{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.z %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// Now the lambda instances ...

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test2aZN6test2b
// CHECK-SAME:      [[LAM2A]](%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN6test2bcl{{.*}}_(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN6test2bcl{{.*}}_(%[[VAL_3]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test2a_cRZN6test2c
// CHECK-SAME:      [[LAM2C]](%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN6test2c{{.*}}_(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN6test2c{{.*}}_(%[[VAL_3]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test4x2Z
// CHECK-SAME:      [[LAM42a]](%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x4cl{{.*}}_(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x4cl{{.*}}_(%[[VAL_3]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_test4x2R
// CHECK-SAME:      [[LAM42b]](%[[VAL_0:.*]]: !cc.callable<(!quake.ref) -> ()>{{.*}}, %[[VAL_1:.*]]: !quake.veq<?>{{.*}}) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x8cl{{.*}}_(%[[VAL_2]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_1]][1] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__ZN7test4x8cl{{.*}}_(%[[VAL_3]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }
