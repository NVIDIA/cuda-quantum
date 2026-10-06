/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck --check-prefixes=CHECK,ALIVE %s
// RUN: cudaq-quake %s | cudaq-opt -erase-noise | FileCheck --check-prefixes=CHECK,DEAD %s
// RUN: cudaq-quake %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

#include <cudaq.h>

struct SantaKraus : public cudaq::kraus_channel {
  constexpr static std::size_t num_parameters = 0;
  constexpr static std::size_t num_targets = 2;
  static std::size_t get_key() { return (std::size_t)&get_key; }
  SantaKraus() {}
};

struct testApplyNoise {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    cudaq::apply_noise<SantaKraus>(q0, q1);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testApplyNoise() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// ALIVE:           quake.apply_noise @_ZN5cudaq11apply_noiseI{{.*}}SantaKraus{{.*}}() %[[VAL_0]], %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// DEAD-NOT:        quake.apply_noise
// CHECK:           return
// CHECK:         }

// QIR-LABEL: define void @__nvqpp__mlirgen__testApplyNoise() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// QIR:         %[[VAL_1:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// QIR:         tail call void @_ZN5cudaq11apply_noiseI{{.*}}SantaKraus{{.*}}(ptr %[[VAL_0]], ptr %[[VAL_1]])
// QIR:         tail call void @__quantum__rt__qubit_release(ptr %[[VAL_0]])
// QIR:         tail call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
// QIR:         ret void
// QIR:       }
// clang-format on

struct SarahKraus : public cudaq::kraus_channel {
  constexpr static std::size_t num_parameters = 2;
  constexpr static std::size_t num_targets = 1;
  static std::size_t get_key() { return (std::size_t)&get_key; }
  SarahKraus(double dip, float around) {}
};

struct testApplyMoreNoise {
  void operator()() __qpu__ {
    double d = 4.0;
    float f = 5.0f;
    cudaq::qubit q0;
    cudaq::apply_noise<SarahKraus>(d, f, q0);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testApplyMoreNoise()
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<f32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// ALIVE:           quake.apply_noise @_ZN5cudaq11apply_noise{{.*}}SarahKraus{{.*}}(%[[VAL_2]], %[[VAL_3]]) %[[VAL_4]] : (!cc.ptr<f64>, !cc.ptr<f32>, !quake.ref) -> ()
// DEAD-NOT:        quake.apply_noise
// CHECK:           return
// CHECK:         }

// QIR-LABEL: define void @__nvqpp__mlirgen__testApplyMoreNoise() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate()
// QIR:         %[[VAL_1:.*]] = alloca double, align 8
// QIR:         store double 4.000000e+00, ptr %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = alloca float, align 4
// QIR:         store float 5.000000e+00, ptr %[[VAL_2]], align 4
// QIR:         call void @_ZN5cudaq11apply_noise{{.*}}SarahKraus{{.*}}(ptr nonnull %[[VAL_1]], ptr nonnull %[[VAL_2]], ptr %[[VAL_0]])
// QIR:         call void @__quantum__rt__qubit_release(ptr %[[VAL_0]])
// QIR:         ret void
// QIR:       }
// clang-format on
