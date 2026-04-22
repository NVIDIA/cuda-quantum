/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt -lambda-lifting=constant-prop=1 -apply-op-specialization | FileCheck %s
// clang-format on

#include <cmath>
#include <cudaq.h>
#include <numbers>

__qpu__ void reflect_about_uniform(cudaq::qvector<> &qs) {
  auto ctrlQubits = qs.front(qs.size() - 1);
  auto &lastQubit = qs.back();

  cudaq::compute_action(
      [&]() {
        h(qs);
        x(qs);
      },
      [&]() { z<cudaq::ctrl>(ctrlQubits, lastQubit); });
}

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, CallableKernel &&oracle,
                          const long target_state) {
    int n_iterations = round(0.25 * M_PI * sqrt(2 ^ n_qubits));

    cudaq::qvector qs(n_qubits);
    h(qs);
    for (int i = 0; i < n_iterations; i++) {
      oracle(target_state, qs);
      reflect_about_uniform(qs);
    }
    mz(qs);
  }
};

struct oracle {
  void operator()(const long target_state, cudaq::qvector<> &qs) __qpu__ {
    cudaq::compute_action(
        [&]() {
          for (int i = 1; i <= qs.size(); ++i) {
            auto target_bit_set = (1 << (qs.size() - i)) & target_state;
            if (!target_bit_set)
              x(qs[i - 1]);
          }
        },
        [&]() {
          auto ctrlQubits = qs.front(qs.size() - 1);
          z<cudaq::ctrl>(ctrlQubits, qs.back());
        });
  }
};

int main(int argc, char *argv[]) {
  auto secret = 1 < argc ? strtol(argv[1], NULL, 2) : 0b1011;
  auto counts = cudaq::sample(run_grover{}, 4, oracle{}, secret);
  printf("Found string %s\n", counts.most_probable().c_str());
  return 0;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_reflect_about_uniform._Z21reflect_about_uniformRN5cudaq7qvectorILm2EEE(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_4]], %[[VAL_1]] : i64
// CHECK:           %[[VAL_6:.*]] = quake.subveq %[[VAL_0]], 0, %[[VAL_5]] : (!quake.veq<?>, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           quake.apply @__nvqpp__lifted.lambda.[[C0:[0-9]*]] %[[VAL_0]] : (!quake.veq<?>) -> ()
// CHECK:           quake.apply @__nvqpp__lifted.lambda.[[C1:[0-9]*]] %[[VAL_6]], %[[VAL_9]] : (!quake.veq<?>, !quake.ref) -> ()
// CHECK:           quake.apply<adj> @__nvqpp__lifted.lambda.[[C0]] %[[VAL_0]] : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__oracle(
// CHECK-SAME:      %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_7:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<i64>
// CHECK:           quake.apply @__nvqpp__lifted.lambda.[[C2:[0-9]*]] %[[VAL_1]], %[[VAL_7]] : (!quake.veq<?>, !cc.ptr<i64>) -> ()
// CHECK:           quake.apply @__nvqpp__lifted.lambda.[[C3:[0-9]*]] %[[VAL_1]] : (!quake.veq<?>) -> ()
// CHECK:           quake.apply<adj> @__nvqpp__lifted.lambda.[[C2]] %[[VAL_1]], %[[VAL_7]] : (!quake.veq<?>, !cc.ptr<i64>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_run_groveroracle._ZN10run_groverclI6oracleEEDaiOT_l(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: !cc.struct<"oracle" {} [8,1]>, %[[VAL_2:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.78539816339744828 : f64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_9:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.struct<"oracle" {} [8,1]>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_10]] : !cc.ptr<!cc.struct<"oracle" {} [8,1]>>
// CHECK:           %[[VAL_11:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_2]], %[[VAL_11]] : !cc.ptr<i64>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = arith.xori %[[VAL_12]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_38:.*]] = cc.cast signed %[[VAL_13]] : (i32) -> f64
// CHECK:           %[[VAL_14:.*]] = math.sqrt %[[VAL_38]] : f64
// CHECK:           %[[VAL_15:.*]] = arith.mulf %[[VAL_14]], %[[VAL_3]] : f64
// CHECK:           %[[VAL_16:.*]] = math.round %[[VAL_15]] : f64
// CHECK:           %[[VAL_17:.*]] = cc.cast signed %[[VAL_16]] : (f64) -> i32
// CHECK:           %[[VAL_18:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = cc.cast signed %[[VAL_19]] : (i32) -> i64
// CHECK:           %[[VAL_21:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_20]] : i64]
// CHECK:           %[[VAL_22:.*]] = quake.veq_size %[[VAL_21]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_23:.*]] = cc.loop while ((%[[VAL_24:.*]] = %[[VAL_7]]) -> (i64)) {
// CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_22]] : i64
// CHECK:             cc.condition %[[VAL_25]](%[[VAL_24]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_26:.*]]: i64):
// CHECK:             %[[VAL_27:.*]] = quake.extract_ref %[[VAL_21]]{{\[}}%[[VAL_26]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_27]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_26]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i64):
// CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_6]] : i64
// CHECK:             cc.continue %[[VAL_29]] : i64
// CHECK:           } {invariant}
// CHECK:           cc.scope {
// CHECK:             %[[VAL_30:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_30]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_31:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i32>
// CHECK:               %[[VAL_32:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:               %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               cc.condition %[[VAL_33]]
// CHECK:             } do {
// CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i64>
// CHECK:               func.call @__nvqpp__mlirgen__oracle(%[[VAL_34]], %[[VAL_21]]) : (i64, !quake.veq<?>) -> ()
// CHECK:               func.call @__nvqpp__mlirgen__function_reflect_about_uniform._Z21reflect_about_uniformRN5cudaq7qvectorILm2EEE(%[[VAL_21]]) : (!quake.veq<?>) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_35:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i32>
// CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_4]] : i32
// CHECK:               cc.store %[[VAL_36]], %[[VAL_30]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = quake.mz %[[VAL_21]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.0(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.0(
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.1(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.1(
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.2(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.2(
// CHECK-DAG:   func.func private @__nvqpp__callable.thunk.lambda.3(
// CHECK-DAG:   func.func private @__nvqpp__lifted.lambda.3(
// clang-format on
