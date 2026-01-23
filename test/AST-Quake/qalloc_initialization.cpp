/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake -D CUDAQ_SIMULATION_SCALAR_FP64 %s | cudaq-opt | FileCheck %s
// RUN: cudaq-quake -D CUDAQ_SIMULATION_SCALAR_FP64 %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

// Test various flavors of qubits declared with initial state information.

#include <cudaq.h>

struct Vanilla {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v{cudaq::state{0., 1., 1., 0.}};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Vanilla() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.array<f64 x 4>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_8]] : !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_5]][3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<f64>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_6]], %[[VAL_4]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct VanillaBean {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v = cudaq::state{0., 1., 1., 0.};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VanillaBean() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.array<f64 x 4>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_8]] : !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_5]][3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<f64>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_6]], %[[VAL_4]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct Cherry {
  std::vector<bool> operator()() __qpu__ {
    using namespace std::complex_literals;
    cudaq::qvector v{{std::initializer_list<std::complex<double>>{
        {0.0, 1.0}, {0.6, 0.4}, {1.0, 0.0}, {0.0, 0.0}}}};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Cherry() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [6.000000e-01, 4.000000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<!cc.array<complex<f64> x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_7]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_7]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_12]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_13:.*]] = quake.create_state %[[VAL_8]], %[[VAL_6]] : (!cc.ptr<!cc.array<complex<f64> x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_14:.*]] = quake.get_number_of_qubits %[[VAL_13]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_15:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_14]] : i64]
// CHECK:           %[[VAL_16:.*]] = quake.init_state %[[VAL_15]], %[[VAL_13]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_13]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_17:.*]] = quake.veq_size %[[VAL_16]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct MooseTracks {
  std::vector<bool> operator()() __qpu__ {
    using namespace std::complex_literals;
    cudaq::qvector v{
        {std::complex<double>{0.0, 1.0}, std::complex<double>{0.75, 0.25},
         std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0}}};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MooseTracks() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [7.500000e-01, 2.500000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<!cc.array<complex<f64> x ?>>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_7]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_7]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_12]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_13:.*]] = quake.create_state %[[VAL_8]], %[[VAL_6]] : (!cc.ptr<!cc.array<complex<f64> x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_14:.*]] = quake.get_number_of_qubits %[[VAL_13]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_15:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_14]] : i64]
// CHECK:           %[[VAL_16:.*]] = quake.init_state %[[VAL_15]], %[[VAL_13]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_13]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_17:.*]] = quake.veq_size %[[VAL_16]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct RockyRoad {
  std::vector<bool> operator()() __qpu__ {
    using namespace std::complex_literals;
    cudaq::qvector v{cudaq::state{0.0 + 1.0i, std::complex<double>{0.8, 0.2},
                                  1.0 + 0.0i, std::complex<double>{0.0, 0.0}}};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RockyRoad() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [8.000000e-01, 2.000000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f[[F80:[0-9]*]]
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f[[F80]]
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = call @_ZNS{{.*}}(%[[VAL_7]]) : (f[[F80]]) -> complex<f64>
// CHECK:           %[[VAL_11:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = call @_Z{{.*}}(%[[VAL_9]], %[[VAL_11]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_13:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_6]], %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_14:.*]] = call @_ZNS{{.*}}(%[[VAL_5]]) : (f[[F80]]) -> complex<f64>
// CHECK:           %[[VAL_15:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_16:.*]] = call @_Z{{.*}}(%[[VAL_13]], %[[VAL_15]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<!cc.array<complex<f64> x ?>>
// CHECK:           %[[VAL_19:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_12]], %[[VAL_19]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_20]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_17]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_16]], %[[VAL_21]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_17]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_22]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_23:.*]] = quake.create_state %[[VAL_18]], %[[VAL_4]] : (!cc.ptr<!cc.array<complex<f64> x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_24:.*]] = quake.get_number_of_qubits %[[VAL_23]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_25:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_24]] : i64]
// CHECK:           %[[VAL_26:.*]] = quake.init_state %[[VAL_25]], %[[VAL_23]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_23]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_27:.*]] = quake.veq_size %[[VAL_26]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

std::vector<double> getTwoTimesRank();

struct Pistachio {
  bool operator()() __qpu__ {
    cudaq::qvector v{getTwoTimesRank()};
    h(v);
    return mz(v[0]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Pistachio() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 8 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = call @_Z15getTwoTimesRankv() : () -> !cc.stdvec<f64>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_7:.*]] = cc.alloca f64{{\[}}%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<f64>) -> !cc.ptr<i8>
// CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_8]], %[[VAL_9]], %[[VAL_6]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_10]], %[[VAL_5]] : (!cc.ptr<f64>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : i1
// CHECK:         }
// clang-format on

struct ChocolateMint {
  bool operator()() __qpu__ {
    cudaq::qvector v(getTwoTimesRank());
    h(v);
    return mz(v[0]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ChocolateMint() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 8 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = call @_Z15getTwoTimesRankv() : () -> !cc.stdvec<f64>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_7:.*]] = cc.alloca f64{{\[}}%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<f64>) -> !cc.ptr<i8>
// CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_8]], %[[VAL_9]], %[[VAL_6]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_10]], %[[VAL_5]] : (!cc.ptr<f64>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : i1
// CHECK:         }
// clang-format on

std::vector<std::complex<double>> getComplexInit();

struct Neapolitan {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v{getComplexInit()};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Neapolitan() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = call @_Z14getComplexInitv() : () -> !cc.stdvec<complex<f64>>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_7:.*]] = cc.alloca complex<f64>{{\[}}%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<complex<f64>>) -> !cc.ptr<i8>
// CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_8]], %[[VAL_9]], %[[VAL_6]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_10]], %[[VAL_5]] : (!cc.ptr<complex<f64>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct ButterPecan {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qvector v(getComplexInit());
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ButterPecan() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = call @_Z14getComplexInitv() : () -> !cc.stdvec<complex<f64>>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_5]], %[[VAL_0]] : i64
// CHECK:           %[[VAL_7:.*]] = cc.alloca complex<f64>{{\[}}%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<complex<f64>>) -> !cc.ptr<i8>
// CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_8]], %[[VAL_9]], %[[VAL_6]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = quake.create_state %[[VAL_10]], %[[VAL_5]] : (!cc.ptr<complex<f64>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_12:.*]] = quake.get_number_of_qubits %[[VAL_11]] : (!cc.ptr<!quake.state>) -> i64
// CHECK:           %[[VAL_13:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_12]] : i64]
// CHECK:           %[[VAL_14:.*]] = quake.init_state %[[VAL_13]], %[[VAL_11]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// CHECK:           quake.delete_state %[[VAL_11]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// CHECK:           return %{{.*}} : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

__qpu__ auto Strawberry() {
  cudaq::qubit q(cudaq::state{0., 1.});
  return mz(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<f64 x 2>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = quake.create_state %[[VAL_4]], %[[VAL_2]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_9:.*]] = quake.init_state %[[VAL_8]], %[[VAL_7]] : (!quake.ref, !cc.ptr<!quake.state>) -> !quake.veq<1>
// CHECK:           quake.delete_state %[[VAL_7]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_10:.*]] = quake.extract_ref %[[VAL_9]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_11:.*]] = quake.mz %[[VAL_10]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_12:.*]] = quake.discriminate %[[VAL_11]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_12]] : i1
// CHECK:         }
// clang-format on

#if 0
// The ket syntax is not yet provided in the headers.
__qpu__ auto GoldRibbon() {
  cudaq::qubit q = cudaq::ket::one;
  return mz(q);
}
#endif

__qpu__ bool Peppermint() {
  cudaq::qubit q{{M_SQRT1_2, M_SQRT1_2}};
  return mz(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_Peppermint._Z10Peppermintv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0.70710678118654757 : f64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = cc.alloca !cc.array<f64 x 2>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = quake.create_state %[[VAL_3]], %[[VAL_1]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<!quake.state>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.init_state %[[VAL_7]], %[[VAL_6]] : (!quake.ref, !cc.ptr<!quake.state>) -> !quake.veq<1>
// CHECK:           quake.delete_state %[[VAL_6]] : !cc.ptr<!quake.state>
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_9]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_11:.*]] = quake.discriminate %[[VAL_10]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_11]] : i1
// CHECK:         }
// clang-format on

//===----------------------------------------------------------------------===//
//
// QIR checks
//
//===----------------------------------------------------------------------===//

// clang-format off
// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Vanilla() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x double], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 0
// QIR:         store double 0.000000e+00, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 1
// QIR:         store double 1.000000e+00, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 2
// QIR:         store double 1.000000e+00, double* %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 3
// QIR:         store double 0.000000e+00, double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast [4 x double]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_6:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_5]], i64 4)
// QIR:         %[[VAL_7:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_7]], i8** %[[VAL_6]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_6]])
// QIR:        %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__VanillaBean() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x double], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 0
// QIR:         store double 0.000000e+00, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 1
// QIR:         store double 1.000000e+00, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 2
// QIR:         store double 1.000000e+00, double* %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x double], [4 x double]* %[[VAL_0]], i64 0, i64 3
// QIR:         store double 0.000000e+00, double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast [4 x double]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_6:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_5]], i64 4)
// QIR:         %[[VAL_7:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_7]], i8** %[[VAL_6]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_6]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Cherry() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// QIR:         store double 0.000000e+00, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// QIR:         store double 1.000000e+00, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         store double 6.000000e-01, double* %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 4.000000e-01, double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 0
// QIR:         store double 1.000000e+00, double* %[[VAL_5]], align 8
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 1
// QIR:         %[[VAL_7:.*]] = bitcast [4 x { double, double }]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_8:.*]] = bitcast double* %[[VAL_6]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %[[VAL_8]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_9:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_7]], i64 4)
// QIR:         %[[VAL_10:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_9]])
// QIR:         %[[VAL_11:.*]] = call %[[VAL_12:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_10]], i8** %[[VAL_9]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_9]])
// QIR:         %[[VAL_13:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_12]]* %[[VAL_11]])
// QIR:         %[[VAL_14:.*]] = icmp sgt i64 %[[VAL_13]], 0
// QIR:         br i1 %[[VAL_14]], label %[[VAL_15:.*]], label %[[VAL_16:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__MooseTracks() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// QIR:         store double 0.000000e+00, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// QIR:         store double 1.000000e+00, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         store double 7.500000e-01, double* %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 2.500000e-01, double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 0
// QIR:         store double 1.000000e+00, double* %[[VAL_5]], align 8
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 1
// QIR:         %[[VAL_7:.*]] = bitcast [4 x { double, double }]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_8:.*]] = bitcast double* %[[VAL_6]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(24) %[[VAL_8]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_9:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_7]], i64 4)
// QIR:         %[[VAL_10:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_9]])
// QIR:         %[[VAL_11:.*]] = call %[[VAL_12:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_10]], i8** %[[VAL_9]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_9]])
// QIR:         %[[VAL_13:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_12]]* %[[VAL_11]])
// QIR:         %[[VAL_14:.*]] = icmp sgt i64 %[[VAL_13]], 0
// QIR:         br i1 %[[VAL_14]], label %[[VAL_15:.*]], label %[[VAL_16:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__RockyRoad() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca double, align 8
// QIR:         store double 0.000000e+00, double* %[[VAL_0]], align 8
// QIR:         %[[VAL_2:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_3:.*]] = extractvalue { double, double } %{{.*}}, 0
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 0
// QIR:         store double %[[VAL_3]], double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = extractvalue { double, double } %{{.*}}, 1
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 1
// QIR:         store double %[[VAL_5]], double* %[[VAL_6]], align 8
// QIR:         %[[VAL_7:.*]] = call { double, double } @_Z{{.*}}(double* nonnull %[[VAL_0]], { double, double }* nonnull %[[VAL_2]])
// QIR:         %[[VAL_8:.*]] = alloca double, align 8
// QIR:         store double 1.000000e+00, double* %[[VAL_8]], align 8
// QIR:         %[[VAL_10:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_11:.*]] = extractvalue { double, double } %{{.*}}, 0
// QIR:         %[[VAL_12:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 0
// QIR:         store double %[[VAL_11]], double* %[[VAL_12]], align 8
// QIR:         %[[VAL_13:.*]] = extractvalue { double, double } %{{.*}}, 1
// QIR:         %[[VAL_14:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 1
// QIR:         store double %[[VAL_13]], double* %[[VAL_14]], align 8
// QIR:         %[[VAL_15:.*]] = call { double, double } @_Z{{.*}}(double* nonnull %[[VAL_8]], { double, double }* nonnull %[[VAL_10]])
// QIR:         %[[VAL_16:.*]] = alloca [4 x { double, double }], align 8
// QIR:         %[[VAL_17:.*]] = extractvalue { double, double } %[[VAL_7]], 0
// QIR:         %[[VAL_18:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 0, i32 0
// QIR:         store double %[[VAL_17]], double* %[[VAL_18]], align 8
// QIR:         %[[VAL_19:.*]] = extractvalue { double, double } %[[VAL_7]], 1
// QIR:         %[[VAL_20:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 0, i32 1
// QIR:         store double %[[VAL_19]], double* %[[VAL_20]], align 8
// QIR:         %[[VAL_21:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 1, i32 0
// QIR:         store double 8.000000e-01, double* %[[VAL_21]], align 8
// QIR:         %[[VAL_22:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 1, i32 1
// QIR:         store double 2.000000e-01, double* %[[VAL_22]], align 8
// QIR:         %[[VAL_23:.*]] = extractvalue { double, double } %[[VAL_15]], 0
// QIR:         %[[VAL_24:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 2, i32 0
// QIR:         store double %[[VAL_23]], double* %[[VAL_24]], align 8
// QIR:         %[[VAL_25:.*]] = extractvalue { double, double } %[[VAL_15]], 1
// QIR:         %[[VAL_26:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 2, i32 1
// QIR:         store double %[[VAL_25]], double* %[[VAL_26]], align 8
// QIR:         %[[VAL_27:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 3, i32 0
// QIR:         %[[VAL_28:.*]] = bitcast [4 x { double, double }]* %[[VAL_16]] to i8*
// QIR:         %[[VAL_29:.*]] = bitcast double* %[[VAL_27]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 8 dereferenceable(16) %[[VAL_29]], i8 0, i64 16, i1 false)
// QIR:         %[[VAL_30:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_28]], i64 4)
// QIR:         %[[VAL_31:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_30]])
// QIR:         %[[VAL_32:.*]] = call %[[VAL_33:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_31]], i8** %[[VAL_30]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_30]])
// QIR:         %[[VAL_34:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_33]]* %[[VAL_32]])
// QIR:         %[[VAL_35:.*]] = icmp sgt i64 %[[VAL_34]], 0
// QIR:         br i1 %[[VAL_35]], label %[[VAL_36:.*]], label %[[VAL_37:.*]]

// QIR-LABEL: define i1 @__nvqpp__mlirgen__Pistachio() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast double* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast double* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[VAL_5]], i8* align 1 %[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_5]], i64 %
// QIR:         %[[VAL_9:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_8]])
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_9]], i8** %[[VAL_8]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_11]]* %[[VAL_10]])
// QIR:         %[[VAL_13:.*]] = icmp sgt i64 %[[VAL_12]], 0
// QIR:         br i1 %[[VAL_13]], label %[[VAL_14:.*]], label %[[VAL_15:.*]]

// QIR-LABEL: define i1 @__nvqpp__mlirgen__ChocolateMint() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast double* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast double* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[VAL_5]], i8* align 1 %[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_5]], i64 %
// QIR:         %[[VAL_9:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_8]])
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_9]], i8** %[[VAL_8]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_11]]* %[[VAL_10]])
// QIR:         %[[VAL_13:.*]] = icmp sgt i64 %[[VAL_12]], 0
// QIR:         br i1 %[[VAL_13]], label %[[VAL_14:.*]], label %[[VAL_15:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Neapolitan() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast { double, double }* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast { double, double }* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[VAL_5]], i8* align 1 %[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_5]], i64 %
// QIR:         %[[VAL_9:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_8]])
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_9]], i8** %[[VAL_8]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_11]]* %[[VAL_10]])
// QIR:         %[[VAL_13:.*]] = icmp sgt i64 %[[VAL_12]], 0
// QIR:         br i1 %[[VAL_13]], label %[[VAL_14:.*]], label %[[VAL_15:.*]]

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__ButterPecan() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]], align 8
// QIR:         %[[VAL_5:.*]] = bitcast { double, double }* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast { double, double }* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 %[[VAL_5]], i8* align 1 %[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_complex_f64(i8* nonnull %[[VAL_5]], i64 %
// QIR:         %[[VAL_9:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(i8** %[[VAL_8]])
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_9]], i8** %[[VAL_8]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_11]]* %[[VAL_10]])
// QIR:         %[[VAL_13:.*]] = icmp sgt i64 %[[VAL_12]], 0
// QIR:         br i1 %[[VAL_13]], label %[[VAL_14:.*]], label %[[VAL_15:.*]]

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [2 x double], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[VAL_0]], i64 0, i64 0
// QIR:         store double 0.000000e+00, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[VAL_0]], i64 0, i64 1
// QIR:         store double 1.000000e+00, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = bitcast [2 x double]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_4:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_3]], i64 2)
// QIR:         %[[VAL_5:.*]] = call %[[VAL_6:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 1, i8** %[[VAL_4]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_4]])
// QIR:         %[[VAL_7:.*]] = call %[[VAL_8:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_6]]* %[[VAL_5]], i64 0)
// QIR:         %[[VAL_9:.*]] = load %[[VAL_8]]*, %[[VAL_8]]** %[[VAL_7]], align 8
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__qis__mz(%[[VAL_8]]* %[[VAL_9]])
// QIR:         %[[VAL_12:.*]] = bitcast %[[VAL_11]]* %[[VAL_10]] to i1*
// QIR:         %[[VAL_13:.*]] = load i1, i1* %[[VAL_12]], align 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_6]]* %[[VAL_5]])
// QIR:         ret i1 %[[VAL_13]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Peppermint._Z10Peppermintv() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [2 x double], align 8
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[VAL_0]], i64 0, i64 0
// QIR:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_1]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [2 x double], [2 x double]* %[[VAL_0]], i64 0, i64 1
// QIR:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = bitcast [2 x double]* %[[VAL_0]] to i8*
// QIR:         %[[VAL_4:.*]] = call i8** @__nvqpp_cudaq_state_createFromData_f64(i8* nonnull %[[VAL_3]], i64 2)
// QIR:         %[[VAL_5:.*]] = call %[[VAL_6:.*]]* @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 1, i8** %[[VAL_4]])
// QIR:         call void @__nvqpp_cudaq_state_delete(i8** %[[VAL_4]])
// QIR:         %[[VAL_7:.*]] = call %[[VAL_8:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_6]]* %[[VAL_5]], i64 0)
// QIR:         %[[VAL_9:.*]] = load %[[VAL_8]]*, %[[VAL_8]]** %[[VAL_7]], align 8
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__qis__mz(%[[VAL_8]]* %[[VAL_9]])
// QIR:         %[[VAL_12:.*]] = bitcast %[[VAL_11]]* %[[VAL_10]] to i1*
// QIR:         %[[VAL_13:.*]] = load i1, i1* %[[VAL_12]], align 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_6]]* %[[VAL_5]])
// QIR:         ret i1 %[[VAL_13]]
// QIR:       }

