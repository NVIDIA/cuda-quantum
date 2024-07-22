/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for recursive std::vector support. Recursive vectors are useful for
// creating a ragged array of values. That is, for any given row, the number of
// columns can be distinct from other rows.

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

void do_something(double);

struct VectorVectorReader {
  void operator()(std::vector<std::vector<double>> theta) __qpu__ {
    for (std::size_t i = 0, N = theta.size(); i < N; ++i) {
      auto &v = theta[i];
      for (std::size_t j = 0, M = v.size(); j < M; ++j)
        do_something(v[j]);
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorVectorReader(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.stdvec<f64>>) attributes
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           cc.scope {
// CHECK:             %[[VAL_3:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:             %[[VAL_4:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<f64>>) -> i64
// CHECK:             %[[VAL_5:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
// CHECK:               %[[VAL_8:.*]] = arith.cmpi ult, %[[VAL_6]], %[[VAL_7]] : i64
// CHECK:               cc.condition %[[VAL_8]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_9:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<f64>>) -> !cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>
// CHECK:                 %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][%[[VAL_9]]] : (!cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>, i64) -> !cc.ptr<!cc.stdvec<f64>>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_12:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_1]], %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                   %[[VAL_13:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.stdvec<f64>>
// CHECK:                   %[[VAL_14:.*]] = cc.stdvec_size %[[VAL_13]] : (!cc.stdvec<f64>) -> i64
// CHECK:                   %[[VAL_15:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<i64>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_16:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_17:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_18:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_17]] : i64
// CHECK:                     cc.condition %[[VAL_18]]
// CHECK:                   } do {
// CHECK:                     %[[VAL_19:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_20:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.stdvec<f64>>
// CHECK:                     %[[VAL_21:.*]] = cc.stdvec_data %[[VAL_20]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:                     %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_21]][%[[VAL_19]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:                     %[[VAL_23:.*]] = cc.load %[[VAL_22]] : !cc.ptr<f64>
// CHECK:                     func.call @_Z12do_somethingd(%[[VAL_23]]) : (f64) -> ()
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_24:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                     cc.store %[[VAL_25]], %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_2]] : i64
// CHECK:               cc.store %[[VAL_27]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

struct TripleVectorReader {
  void operator()(std::vector<std::vector<std::vector<double>>> theta) __qpu__ {
    for (std::size_t i = 0, N = theta.size(); i < N; ++i) {
      auto &v = theta[i];
      for (std::size_t j = 0, M = v.size(); j < M; ++j) {
        auto &w = v[j];
        for (std::size_t k = 0, P = w.size(); k < P; ++k)
          do_something(w[k]);
      }
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__TripleVectorReader(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.stdvec<!cc.stdvec<f64>>>) attributes
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           cc.scope {
// CHECK:             %[[VAL_3:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:             %[[VAL_4:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<!cc.stdvec<f64>>>) -> i64
// CHECK:             %[[VAL_5:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
// CHECK:               %[[VAL_8:.*]] = arith.cmpi ult, %[[VAL_6]], %[[VAL_7]] : i64
// CHECK:               cc.condition %[[VAL_8]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_9:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<!cc.stdvec<f64>>>) -> !cc.ptr<!cc.array<!cc.stdvec<!cc.stdvec<f64>> x ?>>
// CHECK:                 %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][%[[VAL_9]]] : (!cc.ptr<!cc.array<!cc.stdvec<!cc.stdvec<f64>> x ?>>, i64) -> !cc.ptr<!cc.stdvec<!cc.stdvec<f64>>>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_12:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_1]], %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                   %[[VAL_13:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.stdvec<!cc.stdvec<f64>>>
// CHECK:                   %[[VAL_14:.*]] = cc.stdvec_size %[[VAL_13]] : (!cc.stdvec<!cc.stdvec<f64>>) -> i64
// CHECK:                   %[[VAL_15:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<i64>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_16:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_17:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_18:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_17]] : i64
// CHECK:                     cc.condition %[[VAL_18]]
// CHECK:                   } do {
// CHECK:                     cc.scope {
// CHECK:                       %[[VAL_19:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                       %[[VAL_20:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.stdvec<!cc.stdvec<f64>>>
// CHECK:                       %[[VAL_21:.*]] = cc.stdvec_data %[[VAL_20]] : (!cc.stdvec<!cc.stdvec<f64>>) -> !cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>
// CHECK:                       %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_21]][%[[VAL_19]]] : (!cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>, i64) -> !cc.ptr<!cc.stdvec<f64>>
// CHECK:                       cc.scope {
// CHECK:                         %[[VAL_23:.*]] = cc.alloca i64
// CHECK:                         cc.store %[[VAL_1]], %[[VAL_23]] : !cc.ptr<i64>
// CHECK:                         %[[VAL_24:.*]] = cc.load %[[VAL_22]] : !cc.ptr<!cc.stdvec<f64>>
// CHECK:                         %[[VAL_25:.*]] = cc.stdvec_size %[[VAL_24]] : (!cc.stdvec<f64>) -> i64
// CHECK:                         %[[VAL_26:.*]] = cc.alloca i64
// CHECK:                         cc.store %[[VAL_25]], %[[VAL_26]] : !cc.ptr<i64>
// CHECK:                         cc.loop while {
// CHECK:                           %[[VAL_27:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i64>
// CHECK:                           %[[VAL_28:.*]] = cc.load %[[VAL_26]] : !cc.ptr<i64>
// CHECK:                           %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_27]], %[[VAL_28]] : i64
// CHECK:                           cc.condition %[[VAL_29]]
// CHECK:                         } do {
// CHECK:                           %[[VAL_30:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i64>
// CHECK:                           %[[VAL_31:.*]] = cc.load %[[VAL_22]] : !cc.ptr<!cc.stdvec<f64>>
// CHECK:                           %[[VAL_32:.*]] = cc.stdvec_data %[[VAL_31]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:                           %[[VAL_33:.*]] = cc.compute_ptr %[[VAL_32]][%[[VAL_30]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:                           %[[VAL_34:.*]] = cc.load %[[VAL_33]] : !cc.ptr<f64>
// CHECK:                           func.call @_Z12do_somethingd(%[[VAL_34]]) : (f64) -> ()
// CHECK:                           cc.continue
// CHECK:                         } step {
// CHECK:                           %[[VAL_35:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i64>
// CHECK:                           %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_2]] : i64
// CHECK:                           cc.store %[[VAL_36]], %[[VAL_23]] : !cc.ptr<i64>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     }
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_37:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_2]] : i64
// CHECK:                     cc.store %[[VAL_38]], %[[VAL_12]] : !cc.ptr<i64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_39:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
// CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_2]] : i64
// CHECK:               cc.store %[[VAL_40]], %[[VAL_3]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorVectorWriter {
  void operator()(std::vector<std::vector<int>> &theta) __qpu__ {
    for (std::size_t i = 0, N = theta.size(); i < N; ++i) {
      auto &v = theta[i];
      for (std::size_t j = 0, M = v.size(); j < M; ++j)
        v[j] = 42;
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorVectorWriter(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.stdvec<i32>>) attributes
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : i32
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<i32>>) -> i64
// CHECK:             %[[VAL_6:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i64>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi ult, %[[VAL_7]], %[[VAL_8]] : i64
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_10:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_11:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<i32>>) -> !cc.ptr<!cc.array<!cc.stdvec<i32> x ?>>
// CHECK:                 %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_11]][%[[VAL_10]]] : (!cc.ptr<!cc.array<!cc.stdvec<i32> x ?>>, i64) -> !cc.ptr<!cc.stdvec<i32>>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_13:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_1]], %[[VAL_13]] : !cc.ptr<i64>
// CHECK:                   %[[VAL_14:.*]] = cc.load %[[VAL_12]] : !cc.ptr<!cc.stdvec<i32>>
// CHECK:                   %[[VAL_15:.*]] = cc.stdvec_size %[[VAL_14]] : (!cc.stdvec<i32>) -> i64
// CHECK:                   %[[VAL_16:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_17:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_18:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_19:.*]] = arith.cmpi ult, %[[VAL_17]], %[[VAL_18]] : i64
// CHECK:                     cc.condition %[[VAL_19]]
// CHECK:                   } do {
// CHECK:                     %[[VAL_20:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_21:.*]] = cc.load %[[VAL_12]] : !cc.ptr<!cc.stdvec<i32>>
// CHECK:                     %[[VAL_22:.*]] = cc.stdvec_data %[[VAL_21]] : (!cc.stdvec<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:                     %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_22]][%[[VAL_20]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:                     cc.store %[[VAL_3]], %[[VAL_23]] : !cc.ptr<i32>
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_24:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_2]] : i64
// CHECK:                     cc.store %[[VAL_25]], %[[VAL_13]] : !cc.ptr<i64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_2]] : i64
// CHECK:               cc.store %[[VAL_27]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorVectorBilingual {
  void operator()(std::vector<std::vector<double>> &result,
                  std::vector<std::vector<int>> theta) __qpu__ {
    for (std::size_t i = 0, N = theta.size(); i < N; ++i) {
      auto &v = theta[i];
      auto &r = result[i];
      for (std::size_t j = 0, M = v.size(); j < M; ++j)
        r[j] = v[j];
    }
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorVectorBilingual(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.stdvec<f64>>, %[[VAL_1:.*]]: !cc.stdvec<!cc.stdvec<i32>>)
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_1]] : (!cc.stdvec<!cc.stdvec<i32>>) -> i64
// CHECK:             %[[VAL_6:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i64>
// CHECK:               %[[VAL_9:.*]] = arith.cmpi ult, %[[VAL_7]], %[[VAL_8]] : i64
// CHECK:               cc.condition %[[VAL_9]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_10:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_11:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<!cc.stdvec<i32>>) -> !cc.ptr<!cc.array<!cc.stdvec<i32> x ?>>
// CHECK:                 %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_11]][%[[VAL_10]]] : (!cc.ptr<!cc.array<!cc.stdvec<i32> x ?>>, i64) -> !cc.ptr<!cc.stdvec<i32>>
// CHECK:                 %[[VAL_13:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_14:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.stdvec<f64>>) -> !cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>
// CHECK:                 %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_14]][%[[VAL_13]]] : (!cc.ptr<!cc.array<!cc.stdvec<f64> x ?>>, i64) -> !cc.ptr<!cc.stdvec<f64>>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_16:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_2]], %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                   %[[VAL_17:.*]] = cc.load %[[VAL_12]] : !cc.ptr<!cc.stdvec<i32>>
// CHECK:                   %[[VAL_18:.*]] = cc.stdvec_size %[[VAL_17]] : (!cc.stdvec<i32>) -> i64
// CHECK:                   %[[VAL_19:.*]] = cc.alloca i64
// CHECK:                   cc.store %[[VAL_18]], %[[VAL_19]] : !cc.ptr<i64>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_20:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_21:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_21]] : i64
// CHECK:                     cc.condition %[[VAL_22]]
// CHECK:                   } do {
// CHECK:                     %[[VAL_23:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_24:.*]] = cc.load %[[VAL_15]] : !cc.ptr<!cc.stdvec<f64>>
// CHECK:                     %[[VAL_25:.*]] = cc.stdvec_data %[[VAL_24]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:                     %[[VAL_26:.*]] = cc.compute_ptr %[[VAL_25]][%[[VAL_23]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:                     %[[VAL_27:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_28:.*]] = cc.load %[[VAL_12]] : !cc.ptr<!cc.stdvec<i32>>
// CHECK:                     %[[VAL_29:.*]] = cc.stdvec_data %[[VAL_28]] : (!cc.stdvec<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:                     %[[VAL_30:.*]] = cc.compute_ptr %[[VAL_29]][%[[VAL_27]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:                     %[[VAL_31:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_32:.*]] = cc.cast signed %[[VAL_31]] : (i32) -> f64
// CHECK:                     cc.store %[[VAL_32]], %[[VAL_26]] : !cc.ptr<f64>
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_33:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                     %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : i64
// CHECK:                     cc.store %[[VAL_34]], %[[VAL_16]] : !cc.ptr<i64>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_35:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
// CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_3]] : i64
// CHECK:               cc.store %[[VAL_36]], %[[VAL_4]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

