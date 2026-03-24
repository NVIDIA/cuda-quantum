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
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = complex.constant [6.000000e-01, 4.000000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_4:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_8]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_7]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_13:.*]] = quake.init_state %[[VAL_12]], %[[VAL_7]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<2>
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
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = complex.constant [7.500000e-01, 2.500000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_4:.*]] = complex.constant [0.000000e+00, 1.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_7:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_4]], %[[VAL_8]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_9]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_10]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_7]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_13:.*]] = quake.init_state %[[VAL_12]], %[[VAL_7]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<2>
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
// CHECK-DAG:       %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %{{.*}}, %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = call @_ZNS{{.*}}(%{{.*}}) : (f{{[0-9]+}}) -> complex<f64>
// CHECK:           %[[VAL_11:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = call @_Z{{.*}}(%[[VAL_9]], %[[VAL_11]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_13:.*]] = cc.alloca f64
// CHECK:           cc.store %{{.*}}, %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_14:.*]] = call @_ZNS{{.*}}(%{{.*}}) : (f{{[0-9]+}}) -> complex<f64>
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
// CHECK:           %[[VAL_2:.*]] = call @_Z15getTwoTimesRankv() : () -> !cc.stdvec<f64>
// CHECK:           %[[VAL_30:.*]] = cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_31:.*]] = cc.alloca f64[%{{.*}} : i64]
// CHECK:           %[[VAL_3:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_31]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
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
// CHECK:           %[[VAL_2:.*]] = call @_Z15getTwoTimesRankv() : () -> !cc.stdvec<f64>
// CHECK:           %[[VAL_30:.*]] = cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<f64>) -> i64
// CHECK:           %[[VAL_31:.*]] = cc.alloca f64[%{{.*}} : i64]
// CHECK:           %[[VAL_3:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_31]] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_3]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<f64>) -> !quake.veq<?>
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
// CHECK:           %[[VAL_3:.*]] = call @_Z14getComplexInitv() : () -> !cc.stdvec<complex<f64>>
// CHECK:           %[[VAL_30:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> i64
// CHECK:           %[[VAL_31:.*]] = cc.alloca complex<f64>[%{{.*}}]
// CHECK:           %[[VAL_4:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_31]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>[%[[VAL_4]] : i64]
// CHECK:           %[[VAL_7:.*]] = quake.init_state %[[VAL_6]], %[[VAL_5]] : (!quake.veq<?>, !cc.ptr<complex<f64>>) -> !quake.veq<?>
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
// CHECK:           %[[VAL_3:.*]] = call @_Z14getComplexInitv() : () -> !cc.stdvec<complex<f64>>
// CHECK:           %[[VAL_30:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> i64
// CHECK:           %[[VAL_31:.*]] = cc.alloca complex<f64>[%{{.*}}]
// CHECK:           %[[VAL_4:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_31]] : (!cc.ptr<!cc.array<complex<f64> x ?>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>[%[[VAL_4]] : i64]
// CHECK:           %[[VAL_7:.*]] = quake.init_state %[[VAL_6]], %[[VAL_5]] : (!quake.veq<?>, !cc.ptr<complex<f64>>) -> !quake.veq<?>
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
  cudaq::qubit q{M_SQRT1_2, M_SQRT1_2};
  return mz(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_Peppermint._Z10Peppermintv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = complex.constant [0.70710678118654757, 0.000000e+00] : complex<f64>
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.array<complex<f64> x 2>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_3:.*]] = cc.compute_ptr %[[VAL_1]][1] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_5:.*]] = quake.init_state %[[VAL_4]], %[[VAL_1]] : (!quake.veq<1>, !cc.ptr<!cc.array<complex<f64> x 2>>) -> !quake.veq<1>
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_7:.*]] = quake.mz %[[VAL_6]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_8:.*]] = quake.discriminate %[[VAL_7]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_8]] : i1
// CHECK:         }
// clang-format on

//===----------------------------------------------------------------------===//
//
// QIR checks
//
//===----------------------------------------------------------------------===//

// clang-format off
// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__Vanilla() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x double], align 8
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 16
// QIR:         store double 1.000000e+00, ptr %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 24
// QIR:         store double 0.000000e+00, ptr %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = call ptr @__nvqpp_cudaq_state_createFromData_f64(ptr nonnull %[[VAL_0]], i64 4)
// QIR:         %[[VAL_6:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(ptr %[[VAL_5]])
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_6]], ptr %[[VAL_5]])
// QIR:         call void @__nvqpp_cudaq_state_delete(ptr %[[VAL_5]])
// QIR:         %[[VAL_8:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_7]])
// QIR:         %[[VAL_9:.*]] = icmp sgt i64 %[[VAL_8]], 0
// QIR:         br i1 %[[VAL_9]], label %[[VAL_10:.*]], label %[[VAL_11:.*]]

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__VanillaBean() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [4 x double], align 8
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 16
// QIR:         store double 1.000000e+00, ptr %[[VAL_3]], align 8
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 24
// QIR:         store double 0.000000e+00, ptr %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = call ptr @__nvqpp_cudaq_state_createFromData_f64(ptr nonnull %[[VAL_0]], i64 4)
// QIR:         %[[VAL_6:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(ptr %[[VAL_5]])
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_6]], ptr %[[VAL_5]])
// QIR:         call void @__nvqpp_cudaq_state_delete(ptr %[[VAL_5]])
// QIR:         %[[VAL_8:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_7]])
// QIR:         %[[VAL_9:.*]] = icmp sgt i64 %[[VAL_8]], 0
// QIR:         br i1 %[[VAL_9]], label %[[VAL_10:.*]], label %[[VAL_11:.*]]

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__Cherry()
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }]
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_1]]
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 16
// QIR:         store double 6.000000e-01, ptr %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 24
// QIR:         store double 4.000000e-01, ptr %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 32
// QIR:         store double 1.000000e+00, ptr %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 40
// QIR:         call void @llvm.memset.p0.i64(ptr {{.*}}%[[VAL_5]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_6:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, ptr {{.*}}%[[VAL_0]])
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 0)
// QIR:         %[[VAL_8:.*]] = load ptr, ptr %[[VAL_7]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_8]])
// QIR:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 1)
// QIR:         %[[VAL_10:.*]] = load ptr, ptr %[[VAL_9]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_10]])
// QIR:         %[[VAL_11:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = load i1, ptr %[[VAL_11]]
// QIR:         %[[VAL_13:.*]] = zext i1 %[[VAL_12]] to i8
// QIR:         %[[VAL_14:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_10]])
// QIR:         %[[VAL_15:.*]] = load i1, ptr %[[VAL_14]]
// QIR:         %[[VAL_16:.*]] = zext i1 %[[VAL_15]] to i8
// QIR:         %[[VAL_17:.*]] = call dereferenceable_or_null(2) ptr @malloc(i64 2)
// QIR:         store i8 %[[VAL_13]], ptr %[[VAL_17]]
// QIR:         %[[VAL_18:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_17]], i64 1
// QIR:         store i8 %[[VAL_16]], ptr %[[VAL_18]]
// QIR:         %[[VAL_19:.*]] = insertvalue { ptr, i64 } undef, ptr %[[VAL_17]], 0
// QIR:         %[[VAL_20:.*]] = insertvalue { ptr, i64 } %[[VAL_19]], i64 2, 1
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_6]])
// QIR:         ret { ptr, i64 } %[[VAL_20]]
// QIR:       }

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__MooseTracks()
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }]
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_1]]
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 16
// QIR:         store double 7.500000e-01, ptr %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 24
// QIR:         store double 2.500000e-01, ptr %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 32
// QIR:         store double 1.000000e+00, ptr %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 40
// QIR:         call void @llvm.memset.p0.i64(ptr {{.*}}%[[VAL_5]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_6:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, ptr {{.*}}%[[VAL_0]])
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 0)
// QIR:         %[[VAL_8:.*]] = load ptr, ptr %[[VAL_7]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_8]])
// QIR:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 1)
// QIR:         %[[VAL_10:.*]] = load ptr, ptr %[[VAL_9]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_10]])
// QIR:         %[[VAL_11:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_8]])
// QIR:         %[[VAL_12:.*]] = load i1, ptr %[[VAL_11]]
// QIR:         %[[VAL_13:.*]] = zext i1 %[[VAL_12]] to i8
// QIR:         %[[VAL_14:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_10]])
// QIR:         %[[VAL_15:.*]] = load i1, ptr %[[VAL_14]]
// QIR:         %[[VAL_16:.*]] = zext i1 %[[VAL_15]] to i8
// QIR:         %[[VAL_17:.*]] = call dereferenceable_or_null(2) ptr @malloc(i64 2)
// QIR:         store i8 %[[VAL_13]], ptr %[[VAL_17]]
// QIR:         %[[VAL_18:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_17]], i64 1
// QIR:         store i8 %[[VAL_16]], ptr %[[VAL_18]]
// QIR:         %[[VAL_19:.*]] = insertvalue { ptr, i64 } undef, ptr %[[VAL_17]], 0
// QIR:         %[[VAL_20:.*]] = insertvalue { ptr, i64 } %[[VAL_19]], i64 2, 1
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_6]])
// QIR:         ret { ptr, i64 } %[[VAL_20]]
// QIR:       }

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__RockyRoad() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca double, align 8
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]], align 8
// QIR:         %[[VAL_2:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_3:.*]] = extractvalue { double, double } %{{.*}}, 0
// QIR:         store double %[[VAL_3]], ptr %[[VAL_2]], align 8
// QIR:         %[[VAL_5:.*]] = extractvalue { double, double } %{{.*}}, 1
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_2]], i64 8
// QIR:         store double %[[VAL_5]], ptr %[[VAL_6]], align 8
// QIR:         %[[VAL_7:.*]] = call { double, double } @_Z{{.*}}(ptr nonnull %[[VAL_0]], ptr nonnull %[[VAL_2]])
// QIR:         %[[VAL_8:.*]] = alloca double, align 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_8]], align 8
// QIR:         %[[VAL_10:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_11:.*]] = extractvalue { double, double } %{{.*}}, 0
// QIR:         store double %[[VAL_11]], ptr %[[VAL_10]], align 8
// QIR:         %[[VAL_13:.*]] = extractvalue { double, double } %{{.*}}, 1
// QIR:         %[[VAL_14:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_10]], i64 8
// QIR:         store double %[[VAL_13]], ptr %[[VAL_14]], align 8
// QIR:         %[[VAL_15:.*]] = call { double, double } @_Z{{.*}}(ptr nonnull %[[VAL_8]], ptr nonnull %[[VAL_10]])
// QIR:         %[[VAL_16:.*]] = alloca [4 x { double, double }], align 8
// QIR:         %[[VAL_17:.*]] = extractvalue { double, double } %[[VAL_7]], 0
// QIR:         store double %[[VAL_17]], ptr %[[VAL_16]], align 8
// QIR:         %[[VAL_19:.*]] = extractvalue { double, double } %[[VAL_7]], 1
// QIR:         %[[VAL_20:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 8
// QIR:         store double %[[VAL_19]], ptr %[[VAL_20]], align 8
// QIR:         %[[VAL_21:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 16
// QIR:         store double 8.000000e-01, ptr %[[VAL_21]], align 8
// QIR:         %[[VAL_22:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 24
// QIR:         store double 2.000000e-01, ptr %[[VAL_22]], align 8
// QIR:         %[[VAL_23:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 32
// QIR:         %[[VAL_24:.*]] = extractvalue { double, double } %[[VAL_15]], 0
// QIR:         store double %[[VAL_24]], ptr %[[VAL_23]], align 8
// QIR:         %[[VAL_26:.*]] = extractvalue { double, double } %[[VAL_15]], 1
// QIR:         %[[VAL_27:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 40
// QIR:         store double %[[VAL_26]], ptr %[[VAL_27]], align 8
// QIR:         %[[VAL_28:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_16]], i64 48
// QIR:         call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %[[VAL_28]], i8 0, i64 16, i1 false)
// QIR:         %[[VAL_29:.*]] = call ptr @__nvqpp_cudaq_state_createFromData_complex_f64(ptr nonnull %[[VAL_16]], i64 4)
// QIR:         %[[VAL_30:.*]] = call i64 @__nvqpp_cudaq_state_numberOfQubits(ptr %[[VAL_29]])
// QIR:         %[[VAL_31:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 %[[VAL_30]], ptr %[[VAL_29]])
// QIR:         call void @__nvqpp_cudaq_state_delete(ptr %[[VAL_29]])
// QIR:         %[[VAL_32:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_31]])
// QIR:         %[[VAL_33:.*]] = icmp sgt i64 %[[VAL_32]], 0
// QIR:         br i1 %[[VAL_33]], label %[[VAL_34:.*]], label %[[VAL_35:.*]]

// QIR-LABEL: define i1 @__nvqpp__mlirgen__Pistachio()
// QIR:         %[[VAL_0:.*]] = tail call { ptr, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]]
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_4]], ptr {{.*}}%[[VAL_1]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(ptr %[[VAL_1]])
// QIR:         %[[VAL_7:.*]] = tail call {{.*}} i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false)
// QIR:         %[[VAL_8:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_7]], ptr {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14:.*]], %[[VAL_12]]
// QIR:         %[[VAL_15:.*]] = phi i64 [ %[[VAL_16:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_17:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_15]])
// QIR:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_17]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_19]])
// QIR:         %[[VAL_16]] = add nuw nsw i64 %[[VAL_15]], 1
// QIR:         %[[VAL_20:.*]] = icmp eq i64 %[[VAL_16]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_20]], label %[[VAL_13]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]], %[[VAL_14]]
// QIR:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 0)
// QIR:         %[[VAL_22:.*]] = load ptr, ptr %[[VAL_21]]
// QIR:         %[[VAL_23:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_22]])
// QIR:         %[[VAL_26:.*]] = load i1, ptr %[[VAL_23]]
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_8]])
// QIR:         ret i1 %[[VAL_26]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__ChocolateMint()
// QIR:         %[[VAL_0:.*]] = tail call { ptr, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]]
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_4]], ptr {{.*}}%[[VAL_1]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(ptr %[[VAL_1]])
// QIR:         %[[VAL_7:.*]] = tail call {{.*}} i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false)
// QIR:         %[[VAL_8:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_7]], ptr {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14:.*]], %[[VAL_12]]
// QIR:         %[[VAL_15:.*]] = phi i64 [ %[[VAL_16:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_17:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_15]])
// QIR:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_17]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_19]])
// QIR:         %[[VAL_16]] = add nuw nsw i64 %[[VAL_15]], 1
// QIR:         %[[VAL_20:.*]] = icmp eq i64 %[[VAL_16]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_20]], label %[[VAL_13]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]], %[[VAL_14]]
// QIR:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 0)
// QIR:         %[[VAL_22:.*]] = load ptr, ptr %[[VAL_21]]
// QIR:         %[[VAL_23:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_22]])
// QIR:         %[[VAL_26:.*]] = load i1, ptr %[[VAL_23]]
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_8]])
// QIR:         ret i1 %[[VAL_26]]
// QIR:       }

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__Neapolitan()
// QIR:         %[[VAL_0:.*]] = tail call { ptr, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]]
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_4]], ptr {{.*}}%[[VAL_1]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(ptr %[[VAL_1]])
// QIR:         %[[VAL_7:.*]] = tail call {{.*}} i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false)
// QIR:         %[[VAL_8:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 %[[VAL_7]], ptr {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:
// QIR:         %[[VAL_17:.*]] = phi i64 [ %[[VAL_18:.*]], %{{.*}} ], [ 0, %{{.*}} ]
// QIR:         %[[VAL_19:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_17]])
// QIR:         %[[VAL_21:.*]] = load ptr, ptr %[[VAL_19]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_21]])
// QIR:         %[[VAL_18]] = add nuw nsw i64 %[[VAL_17]], 1
// QIR:         %[[VAL_22:.*]] = icmp eq i64 %[[VAL_18]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_22]], label %{{.*}}, label %{{.*}}
// QIR:       ._crit_edge:
// QIR:         %[[VAL_15:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_16:.*]]
// QIR:       .lr.ph4.preheader:
// QIR:         %[[VAL_24:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_25:.*]]
// QIR:       .lr.ph4:
// QIR:         %[[VAL_26:.*]] = phi i64 [ %[[VAL_27:.*]], %[[VAL_25]] ], [ 0, %{{.*}} ]
// QIR:         %[[VAL_28:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_26]])
// QIR:         %[[VAL_29:.*]] = load ptr, ptr %[[VAL_28]]
// QIR:         %[[VAL_30:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_29]])
// QIR:         %[[VAL_33:.*]] = load i1, ptr %[[VAL_30]]
// QIR:         %[[VAL_34:.*]] = getelementptr i8, ptr %[[VAL_24]], i64 %[[VAL_26]]
// QIR:         %[[VAL_35:.*]] = zext i1 %[[VAL_33]] to i8
// QIR:         store i8 %[[VAL_35]], ptr %[[VAL_34]]
// QIR:         %[[VAL_27]] = add nuw nsw i64 %[[VAL_26]], 1
// QIR:         %[[VAL_36:.*]] = icmp eq i64 %[[VAL_27]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_36]], label %[[VAL_16]], label %[[VAL_25]]
// QIR:       ._crit_edge5:
// QIR:         %[[VAL_37:.*]] = phi ptr [ %[[VAL_15]], %{{.*}} ], [ %[[VAL_24]], %{{.*}} ]
// QIR:         %[[VAL_38:.*]] = call ptr @malloc(i64 %[[VAL_10]])
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_38]], ptr {{.*}}%[[VAL_37]], i64 %[[VAL_10]], i1 false)
// QIR:         %[[VAL_40:.*]] = insertvalue { ptr, i64 } undef, ptr %[[VAL_38]], 0
// QIR:         %[[VAL_41:.*]] = insertvalue { ptr, i64 } %[[VAL_40]], i64 %[[VAL_10]], 1
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_8]])
// QIR:         ret { ptr, i64 } %[[VAL_41]]
// QIR:       }

// QIR-LABEL: define { ptr, i64 } @__nvqpp__mlirgen__ButterPecan()
// QIR:         %[[VAL_0:.*]] = tail call { ptr, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { ptr, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]]
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_4]], ptr {{.*}}%[[VAL_1]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(ptr %[[VAL_1]])
// QIR:         %[[VAL_7:.*]] = tail call {{.*}} i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false)
// QIR:         %[[VAL_8:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 %[[VAL_7]], ptr {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(ptr %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:
// QIR:         %[[VAL_17:.*]] = phi i64 [ %[[VAL_18:.*]], %{{.*}} ], [ 0, %{{.*}} ]
// QIR:         %[[VAL_19:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_17]])
// QIR:         %[[VAL_21:.*]] = load ptr, ptr %[[VAL_19]]
// QIR:         call void @__quantum__qis__h(ptr %[[VAL_21]])
// QIR:         %[[VAL_18]] = add nuw nsw i64 %[[VAL_17]], 1
// QIR:         %[[VAL_22:.*]] = icmp eq i64 %[[VAL_18]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_22]], label %{{.*}}, label %{{.*}}
// QIR:       ._crit_edge:
// QIR:         %[[VAL_15:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_16:.*]]
// QIR:       .lr.ph4.preheader:
// QIR:         %[[VAL_24:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_25:.*]]
// QIR:       .lr.ph4:
// QIR:         %[[VAL_26:.*]] = phi i64 [ %[[VAL_27:.*]], %[[VAL_25]] ], [ 0, %{{.*}} ]
// QIR:         %[[VAL_28:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_8]], i64 %[[VAL_26]])
// QIR:         %[[VAL_29:.*]] = load ptr, ptr %[[VAL_28]]
// QIR:         %[[VAL_30:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_29]])
// QIR:         %[[VAL_33:.*]] = load i1, ptr %[[VAL_30]]
// QIR:         %[[VAL_34:.*]] = getelementptr i8, ptr %[[VAL_24]], i64 %[[VAL_26]]
// QIR:         %[[VAL_35:.*]] = zext i1 %[[VAL_33]] to i8
// QIR:         store i8 %[[VAL_35]], ptr %[[VAL_34]]
// QIR:         %[[VAL_27]] = add nuw nsw i64 %[[VAL_26]], 1
// QIR:         %[[VAL_36:.*]] = icmp eq i64 %[[VAL_27]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_36]], label %[[VAL_16]], label %[[VAL_25]]
// QIR:       ._crit_edge5:
// QIR:         %[[VAL_37:.*]] = phi ptr [ %[[VAL_15]], %{{.*}} ], [ %[[VAL_24]], %{{.*}} ]
// QIR:         %[[VAL_38:.*]] = call ptr @malloc(i64 %[[VAL_10]])
// QIR:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[VAL_38]], ptr {{.*}}%[[VAL_37]], i64 %[[VAL_10]], i1 false)
// QIR:         %[[VAL_40:.*]] = insertvalue { ptr, i64 } undef, ptr %[[VAL_38]], 0
// QIR:         %[[VAL_41:.*]] = insertvalue { ptr, i64 } %[[VAL_40]], i64 %[[VAL_10]], 1
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_8]])
// QIR:         ret { ptr, i64 } %[[VAL_41]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca [2 x double], align 8
// QIR:         store double 0.000000e+00, ptr %[[VAL_0]], align 8
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 1.000000e+00, ptr %[[VAL_2]], align 8
// QIR:         %[[VAL_3:.*]] = call ptr @__nvqpp_cudaq_state_createFromData_f64(ptr nonnull %[[VAL_0]], i64 2)
// QIR:         %[[VAL_4:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(i64 1, ptr %[[VAL_3]])
// QIR:         call void @__nvqpp_cudaq_state_delete(ptr %[[VAL_3]])
// QIR:         %[[VAL_5:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_4]], i64 0)
// QIR:         %[[VAL_6:.*]] = load ptr, ptr %[[VAL_5]], align 8
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = load i1, ptr %[[VAL_7]], align 1
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_4]])
// QIR:         ret i1 %[[VAL_8]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Peppermint._Z10Peppermintv()
// QIR:         %[[VAL_0:.*]] = alloca [2 x { double, double }]
// QIR:         store double 0x3FE6A09E667F3BCD, ptr %[[VAL_0]]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 8
// QIR:         store double 0.000000e+00, ptr %[[VAL_1]]
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 16
// QIR:         store double 0x3FE6A09E667F3BCD, ptr %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds nuw i8, ptr %[[VAL_0]], i64 24
// QIR:         store double 0.000000e+00, ptr %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = call ptr @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 1, ptr {{.*}}%[[VAL_0]])
// QIR:         %[[VAL_5:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_4]], i64 0)
// QIR:         %[[VAL_6:.*]] = load ptr, ptr %[[VAL_5]]
// QIR:         %[[VAL_7:.*]] = call ptr @__quantum__qis__mz(ptr %[[VAL_6]])
// QIR:         %[[VAL_8:.*]] = load i1, ptr %[[VAL_7]]
// QIR:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_4]])
// QIR:         ret i1 %[[VAL_8]]
// QIR:       }

