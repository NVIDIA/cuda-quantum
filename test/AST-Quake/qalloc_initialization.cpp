/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
    cudaq::qvector v = {0., 1., 1., 0.};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Vanilla() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = cc.address_of @__nvqpp__rodata_init_0 : !cc.ptr<!cc.array<f64 x 4>>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_5:.*]] = quake.init_state %[[VAL_4]], %[[VAL_3]] : (!quake.veq<2>, !cc.ptr<!cc.array<f64 x 4>>) -> !quake.veq<2>
// clang-format on

struct Cherry {
  std::vector<bool> operator()() __qpu__ {
    using namespace std::complex_literals;
    cudaq::qvector v = std::initializer_list<std::complex<double>>{
        {0.0, 1.0}, {0.6, 0.4}, {1.0, 0.0}, {0.0, 0.0}};
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
    cudaq::qvector v = {
        std::complex<double>{0.0, 1.0}, std::complex<double>{0.75, 0.25},
        std::complex<double>{1.0, 0.0}, std::complex<double>{0.0, 0.0}};
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
    cudaq::qvector v = {0.0 + 1.0i, std::complex<double>{0.8, 0.2}, 1.0 + 0.0i,
                        std::complex<double>{0.0, 0.0}};
    h(v);
    return mz(v);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__RockyRoad() -> !cc.stdvec<i1> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK-DAG:       %[[VAL_2:.*]] = complex.constant [8.000000e-01, 2.000000e-01] : complex<f64>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f{{[0-9]+}}
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f{{[0-9]+}}
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = call @_ZNSt{{.*}}8literals16complex_literalsli1i{{.*}}Ee(%[[VAL_7]]) : (f{{[0-9]+}}) -> complex<f64>
// CHECK:           %[[VAL_11:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = call @_Z{{.*}}7complexIT_{{.*}}_(%[[VAL_9]], %[[VAL_11]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_13:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_6]], %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_14:.*]] = call @_ZNSt{{.*}}8literals16complex_literalsli1i{{.*}}Ee(%[[VAL_5]]) : (f{{[0-9]+}}) -> complex<f64>
// CHECK:           %[[VAL_15:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_16:.*]] = call @_Z{{.*}}7complexIT_{{.*}}_(%[[VAL_13]], %[[VAL_15]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_17:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_18:.*]] = cc.cast %[[VAL_17]] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_12]], %[[VAL_18]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_17]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_19]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_17]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_16]], %[[VAL_20]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_17]][3] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_21]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_22:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_23:.*]] = quake.init_state %[[VAL_22]], %[[VAL_17]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<2>
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
    cudaq::qvector v = getTwoTimesRank();
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
    cudaq::qvector v = getComplexInit();
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
  cudaq::qubit q = {0., 1.};
  return mz(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = complex.constant [1.000000e+00, 0.000000e+00] : complex<f64>
// CHECK:           %[[VAL_1:.*]] = complex.constant [0.000000e+00, 0.000000e+00] : complex<f64>
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.array<complex<f64> x 2>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_6:.*]] = quake.init_state %[[VAL_5]], %[[VAL_2]] : (!quake.veq<1>, !cc.ptr<!cc.array<complex<f64> x 2>>) -> !quake.veq<1>
// CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_7]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_9:.*]] = quake.discriminate %[[VAL_8]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_9]] : i1
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
  cudaq::qubit q = {M_SQRT1_2, M_SQRT1_2};
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

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Vanilla()
// QIR:         %[[VAL_0:.*]] = tail call %[[VAL_1:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 2, double* {{.*}}getelementptr inbounds ([4 x double], [4 x double]* @__nvqpp__rodata_init_0, i64 0, i64 0))
// QIR:         %[[VAL_2:.*]] = tail call %[[VAL_3:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 0)
// QIR:         %[[VAL_4:.*]] = load %[[VAL_3]]*, %[[VAL_3]]** %[[VAL_2]]
// QIR:         tail call void @__quantum__qis__h(%[[VAL_3]]* %[[VAL_4]])
// QIR:         %[[VAL_5:.*]] = tail call %[[VAL_3]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_1]]* %[[VAL_0]], i64 1)
// QIR:         %[[VAL_6:.*]] = load %[[VAL_3]]*, %[[VAL_3]]** %[[VAL_5]]
// QIR:         tail call void @__quantum__qis__h(%[[VAL_3]]* %[[VAL_6]])
// QIR:         %[[VAL_7:.*]] = tail call %[[VAL_8:.*]]* @__quantum__qis__mz(%[[VAL_3]]* %[[VAL_4]])
// QIR:         %[[VAL_9:.*]] = bitcast %[[VAL_8]]* %[[VAL_7]] to i1*
// QIR:         %[[VAL_10:.*]] = load i1, i1* %[[VAL_9]]
// QIR:         %[[VAL_11:.*]] = zext i1 %[[VAL_10]] to i8
// QIR:         %[[VAL_12:.*]] = tail call %[[VAL_8]]* @__quantum__qis__mz(%[[VAL_3]]* %[[VAL_6]])
// QIR:         %[[VAL_13:.*]] = bitcast %[[VAL_8]]* %[[VAL_12]] to i1*
// QIR:         %[[VAL_14:.*]] = load i1, i1* %[[VAL_13]]
// QIR:         %[[VAL_15:.*]] = zext i1 %[[VAL_14]] to i8
// QIR:         %[[VAL_16:.*]] = tail call dereferenceable_or_null(2) i8* @malloc(i64 2)
// QIR:         store i8 %[[VAL_11]], i8* %[[VAL_16]]
// QIR:         %[[VAL_17:.*]] = getelementptr inbounds i8, i8* %[[VAL_16]], i64 1
// QIR:         store i8 %[[VAL_15]], i8* %[[VAL_17]]
// QIR:         %[[VAL_18:.*]] = bitcast i8* %[[VAL_16]] to i1*
// QIR:         %[[VAL_19:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_18]], 0
// QIR:         %[[VAL_20:.*]] = insertvalue { i1*, i64 } %[[VAL_19]], i64 2, 1
// QIR:         tail call void @__quantum__rt__qubit_release_array(%[[VAL_1]]* %[[VAL_0]])
// QIR:         ret { i1*, i64 } %[[VAL_20]]
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Cherry()
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// QIR:         store double 0.000000e+00, double* %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// QIR:         store double 1.000000e+00, double* %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         store double 6.000000e-01, double* %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 4.000000e-01, double* %[[VAL_5]]
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 0
// QIR:         store double 1.000000e+00, double* %[[VAL_6]]
// QIR:         %[[VAL_7:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 1
// QIR:         %[[VAL_8:.*]] = bitcast double* %[[VAL_7]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* {{.*}}%[[VAL_8]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_9:.*]] = call %[[VAL_10:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, { double, double }* {{.*}}%[[VAL_1]])
// QIR:         %[[VAL_11:.*]] = call %[[VAL_12:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 0)
// QIR:         %[[VAL_13:.*]] = load %[[VAL_12]]*, %[[VAL_12]]** %[[VAL_11]]
// QIR:         call void @__quantum__qis__h(%[[VAL_12]]* %[[VAL_13]])
// QIR:         %[[VAL_14:.*]] = call %[[VAL_12]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 1)
// QIR:         %[[VAL_15:.*]] = load %[[VAL_12]]*, %[[VAL_12]]** %[[VAL_14]]
// QIR:         call void @__quantum__qis__h(%[[VAL_12]]* %[[VAL_15]])
// QIR:         %[[VAL_16:.*]] = call %[[VAL_17:.*]]* @__quantum__qis__mz(%[[VAL_12]]* %[[VAL_13]])
// QIR:         %[[VAL_18:.*]] = bitcast %[[VAL_17]]* %[[VAL_16]] to i1*
// QIR:         %[[VAL_19:.*]] = load i1, i1* %[[VAL_18]]
// QIR:         %[[VAL_20:.*]] = zext i1 %[[VAL_19]] to i8
// QIR:         %[[VAL_21:.*]] = call %[[VAL_17]]* @__quantum__qis__mz(%[[VAL_12]]* %[[VAL_15]])
// QIR:         %[[VAL_22:.*]] = bitcast %[[VAL_17]]* %[[VAL_21]] to i1*
// QIR:         %[[VAL_23:.*]] = load i1, i1* %[[VAL_22]]
// QIR:         %[[VAL_24:.*]] = zext i1 %[[VAL_23]] to i8
// QIR:         %[[VAL_25:.*]] = call dereferenceable_or_null(2) i8* @malloc(i64 2)
// QIR:         store i8 %[[VAL_20]], i8* %[[VAL_25]]
// QIR:         %[[VAL_26:.*]] = getelementptr inbounds i8, i8* %[[VAL_25]], i64 1
// QIR:         store i8 %[[VAL_24]], i8* %[[VAL_26]]
// QIR:         %[[VAL_27:.*]] = bitcast i8* %[[VAL_25]] to i1*
// QIR:         %[[VAL_28:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_27]], 0
// QIR:         %[[VAL_29:.*]] = insertvalue { i1*, i64 } %[[VAL_28]], i64 2, 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_10]]* %[[VAL_9]])
// QIR:         ret { i1*, i64 } %[[VAL_29]]
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__MooseTracks()
// QIR:         %[[VAL_0:.*]] = alloca [4 x { double, double }]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// QIR:         store double 0.000000e+00, double* %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// QIR:         store double 1.000000e+00, double* %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         store double 7.500000e-01, double* %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 2.500000e-01, double* %[[VAL_5]]
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 0
// QIR:         store double 1.000000e+00, double* %[[VAL_6]]
// QIR:         %[[VAL_7:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_0]], i64 0, i64 2, i32 1
// QIR:         %[[VAL_8:.*]] = bitcast double* %[[VAL_7]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* {{.*}}%[[VAL_8]], i8 0, i64 24, i1 false)
// QIR:         %[[VAL_9:.*]] = call %[[VAL_10:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, { double, double }* {{.*}}%[[VAL_1]])
// QIR:         %[[VAL_11:.*]] = call %[[VAL_12:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 0)
// QIR:         %[[VAL_13:.*]] = load %[[VAL_12]]*, %[[VAL_12]]** %[[VAL_11]]
// QIR:         call void @__quantum__qis__h(%[[VAL_12]]* %[[VAL_13]])
// QIR:         %[[VAL_14:.*]] = call %[[VAL_12]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 1)
// QIR:         %[[VAL_15:.*]] = load %[[VAL_12]]*, %[[VAL_12]]** %[[VAL_14]]
// QIR:         call void @__quantum__qis__h(%[[VAL_12]]* %[[VAL_15]])
// QIR:         %[[VAL_16:.*]] = call %[[VAL_17:.*]]* @__quantum__qis__mz(%[[VAL_12]]* %[[VAL_13]])
// QIR:         %[[VAL_18:.*]] = bitcast %[[VAL_17]]* %[[VAL_16]] to i1*
// QIR:         %[[VAL_19:.*]] = load i1, i1* %[[VAL_18]]
// QIR:         %[[VAL_20:.*]] = zext i1 %[[VAL_19]] to i8
// QIR:         %[[VAL_21:.*]] = call %[[VAL_17]]* @__quantum__qis__mz(%[[VAL_12]]* %[[VAL_15]])
// QIR:         %[[VAL_22:.*]] = bitcast %[[VAL_17]]* %[[VAL_21]] to i1*
// QIR:         %[[VAL_23:.*]] = load i1, i1* %[[VAL_22]]
// QIR:         %[[VAL_24:.*]] = zext i1 %[[VAL_23]] to i8
// QIR:         %[[VAL_25:.*]] = call dereferenceable_or_null(2) i8* @malloc(i64 2)
// QIR:         store i8 %[[VAL_20]], i8* %[[VAL_25]]
// QIR:         %[[VAL_26:.*]] = getelementptr inbounds i8, i8* %[[VAL_25]], i64 1
// QIR:         store i8 %[[VAL_24]], i8* %[[VAL_26]]
// QIR:         %[[VAL_27:.*]] = bitcast i8* %[[VAL_25]] to i1*
// QIR:         %[[VAL_28:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_27]], 0
// QIR:         %[[VAL_29:.*]] = insertvalue { i1*, i64 } %[[VAL_28]], i64 2, 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_10]]* %[[VAL_9]])
// QIR:         ret { i1*, i64 } %[[VAL_29]]
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__RockyRoad()
// QIR:         %[[VAL_0:.*]] = alloca double
// QIR:         store double 0.000000e+00, double* %[[VAL_0]]
// QIR:         %[[VAL_1:.*]] = tail call { double, double } @_ZNSt{{.*}}8literals16complex_literalsli1i{{.*}}Ee(
// QIR:         %[[VAL_2:.*]] = alloca { double, double }
// QIR:         %[[VAL_3:.*]] = extractvalue { double, double } %[[VAL_1]], 0
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 0
// QIR:         store double %[[VAL_3]], double* %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = extractvalue { double, double } %[[VAL_1]], 1
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 1
// QIR:         store double %[[VAL_5]], double* %[[VAL_6]]
// QIR:         %[[VAL_7:.*]] = call { double, double } @_Z{{.*}}7complex{{.*}}(double* {{.*}}%[[VAL_0]], { double, double }* {{.*}}%[[VAL_2]])
// QIR:         %[[VAL_8:.*]] = alloca double
// QIR:         store double 1.000000e+00, double* %[[VAL_8]]
// QIR:         %[[VAL_9:.*]] = call { double, double } @_ZNSt{{.*}}8literals16complex_literalsli1i{{.*}}Ee(
// QIR:         %[[VAL_10:.*]] = alloca { double, double }
// QIR:         %[[VAL_11:.*]] = extractvalue { double, double } %[[VAL_9]], 0
// QIR:         %[[VAL_12:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 0
// QIR:         store double %[[VAL_11]], double* %[[VAL_12]]
// QIR:         %[[VAL_13:.*]] = extractvalue { double, double } %[[VAL_9]], 1
// QIR:         %[[VAL_14:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 1
// QIR:         store double %[[VAL_13]], double* %[[VAL_14]]
// QIR:         %[[VAL_15:.*]] = call { double, double } @_Z{{.*}}7complex{{.*}}(double* {{.*}}%[[VAL_8]], { double, double }* {{.*}}%[[VAL_10]])
// QIR:         %[[VAL_16:.*]] = alloca [4 x { double, double }]
// QIR:         %[[VAL_17:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 0
// QIR:         %[[VAL_18:.*]] = extractvalue { double, double } %[[VAL_7]], 0
// QIR:         %[[VAL_19:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 0, i32 0
// QIR:         store double %[[VAL_18]], double* %[[VAL_19]]
// QIR:         %[[VAL_20:.*]] = extractvalue { double, double } %[[VAL_7]], 1
// QIR:         %[[VAL_21:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 0, i32 1
// QIR:         store double %[[VAL_20]], double* %[[VAL_21]]
// QIR:         %[[VAL_22:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 1, i32 0
// QIR:         store double 8.000000e-01, double* %[[VAL_22]]
// QIR:         %[[VAL_23:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 1, i32 1
// QIR:         store double 2.000000e-01, double* %[[VAL_23]]
// QIR:         %[[VAL_24:.*]] = extractvalue { double, double } %[[VAL_15]], 0
// QIR:         %[[VAL_25:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 2, i32 0
// QIR:         store double %[[VAL_24]], double* %[[VAL_25]]
// QIR:         %[[VAL_26:.*]] = extractvalue { double, double } %[[VAL_15]], 1
// QIR:         %[[VAL_27:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 2, i32 1
// QIR:         store double %[[VAL_26]], double* %[[VAL_27]]
// QIR:         %[[VAL_28:.*]] = getelementptr inbounds [4 x { double, double }], [4 x { double, double }]* %[[VAL_16]], i64 0, i64 3, i32 0
// QIR:         %[[VAL_29:.*]] = bitcast double* %[[VAL_28]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* {{.*}}%[[VAL_29]], i8 0, i64 16, i1 false)
// QIR:         %[[VAL_30:.*]] = call %[[VAL_31:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 2, { double, double }* {{.*}}%[[VAL_17]])
// QIR:         %[[VAL_32:.*]] = call %[[VAL_33:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_31]]* %[[VAL_30]], i64 0)
// QIR:         %[[VAL_34:.*]] = load %[[VAL_33]]*, %[[VAL_33]]** %[[VAL_32]]
// QIR:         call void @__quantum__qis__h(%[[VAL_33]]* %[[VAL_34]])
// QIR:         %[[VAL_35:.*]] = call %[[VAL_33]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_31]]* %[[VAL_30]], i64 1)
// QIR:         %[[VAL_36:.*]] = load %[[VAL_33]]*, %[[VAL_33]]** %[[VAL_35]]
// QIR:         call void @__quantum__qis__h(%[[VAL_33]]* %[[VAL_36]])
// QIR:         %[[VAL_37:.*]] = call %[[VAL_38:.*]]* @__quantum__qis__mz(%[[VAL_33]]* %[[VAL_34]])
// QIR:         %[[VAL_39:.*]] = bitcast %[[VAL_38]]* %[[VAL_37]] to i1*
// QIR:         %[[VAL_40:.*]] = load i1, i1* %[[VAL_39]]
// QIR:         %[[VAL_41:.*]] = zext i1 %[[VAL_40]] to i8
// QIR:         %[[VAL_42:.*]] = call %[[VAL_38]]* @__quantum__qis__mz(%[[VAL_33]]* %[[VAL_36]])
// QIR:         %[[VAL_43:.*]] = bitcast %[[VAL_38]]* %[[VAL_42]] to i1*
// QIR:         %[[VAL_44:.*]] = load i1, i1* %[[VAL_43]]
// QIR:         %[[VAL_45:.*]] = zext i1 %[[VAL_44]] to i8
// QIR:         %[[VAL_46:.*]] = call dereferenceable_or_null(2) i8* @malloc(i64 2)
// QIR:         store i8 %[[VAL_41]], i8* %[[VAL_46]]
// QIR:         %[[VAL_47:.*]] = getelementptr inbounds i8, i8* %[[VAL_46]], i64 1
// QIR:         store i8 %[[VAL_45]], i8* %[[VAL_47]]
// QIR:         %[[VAL_48:.*]] = bitcast i8* %[[VAL_46]] to i1*
// QIR:         %[[VAL_49:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_48]], 0
// QIR:         %[[VAL_50:.*]] = insertvalue { i1*, i64 } %[[VAL_49]], i64 2, 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_31]]* %[[VAL_30]])
// QIR:         ret { i1*, i64 } %[[VAL_50]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__Pistachio()
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]]
// QIR:         %[[VAL_5:.*]] = bitcast double* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast double* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_7:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false), !range !1
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_7]], double* {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14:.*]], %[[VAL_12]]
// QIR:         %[[VAL_15:.*]] = phi i64 [ %[[VAL_16:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_17:.*]] = call %[[VAL_18:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_15]])
// QIR:         %[[VAL_19:.*]] = load %[[VAL_18]]*, %[[VAL_18]]** %[[VAL_17]]
// QIR:         call void @__quantum__qis__h(%[[VAL_18]]* %[[VAL_19]])
// QIR:         %[[VAL_16]] = add nuw nsw i64 %[[VAL_15]], 1
// QIR:         %[[VAL_20:.*]] = icmp eq i64 %[[VAL_16]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_20]], label %[[VAL_13]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]], %[[VAL_14]]
// QIR:         %[[VAL_21:.*]] = call %[[VAL_18]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 0)
// QIR:         %[[VAL_22:.*]] = load %[[VAL_18]]*, %[[VAL_18]]** %[[VAL_21]]
// QIR:         %[[VAL_23:.*]] = call %[[VAL_24:.*]]* @__quantum__qis__mz(%[[VAL_18]]* %[[VAL_22]])
// QIR:         %[[VAL_25:.*]] = bitcast %[[VAL_24]]* %[[VAL_23]] to i1*
// QIR:         %[[VAL_26:.*]] = load i1, i1* %[[VAL_25]]
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_9]]* %[[VAL_8]])
// QIR:         ret i1 %[[VAL_26]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__ChocolateMint()
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_1:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 3
// QIR:         %[[VAL_4:.*]] = alloca double, i64 %[[VAL_3]]
// QIR:         %[[VAL_5:.*]] = bitcast double* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast double* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_5]], i8* {{.*}}%[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_7:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false), !range !1
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_7]], double* {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14:.*]], %[[VAL_12]]
// QIR:         %[[VAL_15:.*]] = phi i64 [ %[[VAL_16:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_17:.*]] = call %[[VAL_18:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_15]])
// QIR:         %[[VAL_19:.*]] = load %[[VAL_18]]*, %[[VAL_18]]** %[[VAL_17]]
// QIR:         call void @__quantum__qis__h(%[[VAL_18]]* %[[VAL_19]])
// QIR:         %[[VAL_16]] = add nuw nsw i64 %[[VAL_15]], 1
// QIR:         %[[VAL_20:.*]] = icmp eq i64 %[[VAL_16]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_20]], label %[[VAL_13]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]], %[[VAL_14]]
// QIR:         %[[VAL_21:.*]] = call %[[VAL_18]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 0)
// QIR:         %[[VAL_22:.*]] = load %[[VAL_18]]*, %[[VAL_18]]** %[[VAL_21]]
// QIR:         %[[VAL_23:.*]] = call %[[VAL_24:.*]]* @__quantum__qis__mz(%[[VAL_18]]* %[[VAL_22]])
// QIR:         %[[VAL_25:.*]] = bitcast %[[VAL_24]]* %[[VAL_23]] to i1*
// QIR:         %[[VAL_26:.*]] = load i1, i1* %[[VAL_25]]
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_9]]* %[[VAL_8]])
// QIR:         ret i1 %[[VAL_26]]
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Neapolitan()
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]]
// QIR:         %[[VAL_5:.*]] = bitcast { double, double }* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast { double, double }* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_5]], i8* {{.*}}%[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_7:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false), !range !1
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 %[[VAL_7]], { double, double }* {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       ._crit_edge.thread:                               ; preds = %[[VAL_14:.*]]
// QIR:         %[[VAL_15:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_16:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14]], %[[VAL_12]]
// QIR:         %[[VAL_17:.*]] = phi i64 [ %[[VAL_18:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_19:.*]] = call %[[VAL_20:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_17]])
// QIR:         %[[VAL_21:.*]] = load %[[VAL_20]]*, %[[VAL_20]]** %[[VAL_19]]
// QIR:         call void @__quantum__qis__h(%[[VAL_20]]* %[[VAL_21]])
// QIR:         %[[VAL_18]] = add nuw nsw i64 %[[VAL_17]], 1
// QIR:         %[[VAL_22:.*]] = icmp eq i64 %[[VAL_18]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_22]], label %[[VAL_23:.*]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]]
// QIR:         %[[VAL_24:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br i1 %[[VAL_11]], label %[[VAL_25:.*]], label %[[VAL_16]]
// QIR:       .lr.ph4:                                          ; preds = %[[VAL_23]], %[[VAL_25]]
// QIR:         %[[VAL_26:.*]] = phi i64 [ %[[VAL_27:.*]], %[[VAL_25]] ], [ 0, %[[VAL_23]] ]
// QIR:         %[[VAL_28:.*]] = call %[[VAL_20]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_26]])
// QIR:         %[[VAL_29:.*]] = load %[[VAL_20]]*, %[[VAL_20]]** %[[VAL_28]]
// QIR:         %[[VAL_30:.*]] = call %[[VAL_31:.*]]* @__quantum__qis__mz(%[[VAL_20]]* %[[VAL_29]])
// QIR:         %[[VAL_32:.*]] = bitcast %[[VAL_31]]* %[[VAL_30]] to i1*
// QIR:         %[[VAL_33:.*]] = load i1, i1* %[[VAL_32]]
// QIR:         %[[VAL_34:.*]] = getelementptr i8, i8* %[[VAL_24]], i64 %[[VAL_26]]
// QIR:         %[[VAL_35:.*]] = zext i1 %[[VAL_33]] to i8
// QIR:         store i8 %[[VAL_35]], i8* %[[VAL_34]]
// QIR:         %[[VAL_27]] = add nuw nsw i64 %[[VAL_26]], 1
// QIR:         %[[VAL_36:.*]] = icmp eq i64 %[[VAL_27]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_36]], label %[[VAL_16]], label %[[VAL_25]]
// QIR:       ._crit_edge5:                                     ; preds = %[[VAL_25]], %[[VAL_13]], %[[VAL_23]]
// QIR:         %[[VAL_37:.*]] = phi i8* [ %[[VAL_15]], %[[VAL_13]] ], [ %[[VAL_24]], %[[VAL_23]] ], [ %[[VAL_24]], %[[VAL_25]] ]
// QIR:         %[[VAL_38:.*]] = call i8* @malloc(i64 %[[VAL_10]])
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_38]], i8* {{.*}}%[[VAL_37]], i64 %[[VAL_10]], i1 false)
// QIR:         %[[VAL_39:.*]] = bitcast i8* %[[VAL_38]] to i1*
// QIR:         %[[VAL_40:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_39]], 0
// QIR:         %[[VAL_41:.*]] = insertvalue { i1*, i64 } %[[VAL_40]], i64 %[[VAL_10]], 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_9]]* %[[VAL_8]])
// QIR:         ret { i1*, i64 } %[[VAL_41]]
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__ButterPecan()
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_1:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR:         %[[VAL_3:.*]] = shl i64 %[[VAL_2]], 4
// QIR:         %[[VAL_4:.*]] = alloca { double, double }, i64 %[[VAL_3]]
// QIR:         %[[VAL_5:.*]] = bitcast { double, double }* %[[VAL_4]] to i8*
// QIR:         %[[VAL_6:.*]] = bitcast { double, double }* %[[VAL_1]] to i8*
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_5]], i8* {{.*}}%[[VAL_6]], i64 %[[VAL_3]], i1 false)
// QIR:         tail call void @free(i8* %[[VAL_6]])
// QIR:         %[[VAL_7:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_2]], i1 false), !range !1
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 %[[VAL_7]], { double, double }* {{.*}}%[[VAL_4]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:         %[[VAL_11:.*]] = icmp sgt i64 %[[VAL_10]], 0
// QIR:         br i1 %[[VAL_11]], label %[[VAL_12:.*]], label %[[VAL_13:.*]]
// QIR:       ._crit_edge.thread:                               ; preds = %[[VAL_14:.*]]
// QIR:         %[[VAL_15:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br label %[[VAL_16:.*]]
// QIR:       .lr.ph:                                           ; preds = %[[VAL_14]], %[[VAL_12]]
// QIR:         %[[VAL_17:.*]] = phi i64 [ %[[VAL_18:.*]], %[[VAL_12]] ], [ 0, %[[VAL_14]] ]
// QIR:         %[[VAL_19:.*]] = call %[[VAL_20:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_17]])
// QIR:         %[[VAL_21:.*]] = load %[[VAL_20]]*, %[[VAL_20]]** %[[VAL_19]]
// QIR:         call void @__quantum__qis__h(%[[VAL_20]]* %[[VAL_21]])
// QIR:         %[[VAL_18]] = add nuw nsw i64 %[[VAL_17]], 1
// QIR:         %[[VAL_22:.*]] = icmp eq i64 %[[VAL_18]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_22]], label %[[VAL_23:.*]], label %[[VAL_12]]
// QIR:       ._crit_edge:                                      ; preds = %[[VAL_12]]
// QIR:         %[[VAL_24:.*]] = alloca i8, i64 %[[VAL_10]]
// QIR:         br i1 %[[VAL_11]], label %[[VAL_25:.*]], label %[[VAL_16]]
// QIR:       .lr.ph4:                                          ; preds = %[[VAL_23]], %[[VAL_25]]
// QIR:         %[[VAL_26:.*]] = phi i64 [ %[[VAL_27:.*]], %[[VAL_25]] ], [ 0, %[[VAL_23]] ]
// QIR:         %[[VAL_28:.*]] = call %[[VAL_20]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_9]]* %[[VAL_8]], i64 %[[VAL_26]])
// QIR:         %[[VAL_29:.*]] = load %[[VAL_20]]*, %[[VAL_20]]** %[[VAL_28]]
// QIR:         %[[VAL_30:.*]] = call %[[VAL_31:.*]]* @__quantum__qis__mz(%[[VAL_20]]* %[[VAL_29]])
// QIR:         %[[VAL_32:.*]] = bitcast %[[VAL_31]]* %[[VAL_30]] to i1*
// QIR:         %[[VAL_33:.*]] = load i1, i1* %[[VAL_32]]
// QIR:         %[[VAL_34:.*]] = getelementptr i8, i8* %[[VAL_24]], i64 %[[VAL_26]]
// QIR:         %[[VAL_35:.*]] = zext i1 %[[VAL_33]] to i8
// QIR:         store i8 %[[VAL_35]], i8* %[[VAL_34]]
// QIR:         %[[VAL_27]] = add nuw nsw i64 %[[VAL_26]], 1
// QIR:         %[[VAL_36:.*]] = icmp eq i64 %[[VAL_27]], %[[VAL_10]]
// QIR:         br i1 %[[VAL_36]], label %[[VAL_16]], label %[[VAL_25]]
// QIR:       ._crit_edge5:                                     ; preds = %[[VAL_25]], %[[VAL_13]], %[[VAL_23]]
// QIR:         %[[VAL_37:.*]] = phi i8* [ %[[VAL_15]], %[[VAL_13]] ], [ %[[VAL_24]], %[[VAL_23]] ], [ %[[VAL_24]], %[[VAL_25]] ]
// QIR:         %[[VAL_38:.*]] = call i8* @malloc(i64 %[[VAL_10]])
// QIR:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}%[[VAL_38]], i8* {{.*}}%[[VAL_37]], i64 %[[VAL_10]], i1 false)
// QIR:         %[[VAL_39:.*]] = bitcast i8* %[[VAL_38]] to i1*
// QIR:         %[[VAL_40:.*]] = insertvalue { i1*, i64 } undef, i1* %[[VAL_39]], 0
// QIR:         %[[VAL_41:.*]] = insertvalue { i1*, i64 } %[[VAL_40]], i64 %[[VAL_10]], 1
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_9]]* %[[VAL_8]])
// QIR:         ret { i1*, i64 } %[[VAL_41]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv()
// QIR:         %[[VAL_0:.*]] = alloca [2 x { double, double }]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 0
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         %[[VAL_3:.*]] = bitcast [2 x { double, double }]* %[[VAL_0]] to i8*
// QIR:         call void @llvm.memset.p0i8.i64(i8* {{.*}}%[[VAL_3]], i8 0, i64 16, i1 false)
// QIR:         store double 1.000000e+00, double* %[[VAL_2]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 0.000000e+00, double* %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = call %[[VAL_6:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 1, { double, double }* {{.*}}%[[VAL_1]])
// QIR:         %[[VAL_7:.*]] = call %[[VAL_8:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_6]]* %[[VAL_5]], i64 0)
// QIR:         %[[VAL_9:.*]] = load %[[VAL_8]]*, %[[VAL_8]]** %[[VAL_7]]
// QIR:         %[[VAL_10:.*]] = call %[[VAL_11:.*]]* @__quantum__qis__mz(%[[VAL_8]]* %[[VAL_9]])
// QIR:         %[[VAL_12:.*]] = bitcast %[[VAL_11]]* %[[VAL_10]] to i1*
// QIR:         %[[VAL_13:.*]] = load i1, i1* %[[VAL_12]]
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_6]]* %[[VAL_5]])
// QIR:         ret i1 %[[VAL_13]]
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__function_Peppermint._Z10Peppermintv()
// QIR:         %[[VAL_0:.*]] = alloca [2 x { double, double }]
// QIR:         %[[VAL_1:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 0
// QIR:         %[[VAL_2:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 0
// QIR:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_2]]
// QIR:         %[[VAL_3:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 0, i32 1
// QIR:         store double 0.000000e+00, double* %[[VAL_3]]
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 0
// QIR:         store double 0x3FE6A09E667F3BCD, double* %[[VAL_4]]
// QIR:         %[[VAL_5:.*]] = getelementptr inbounds [2 x { double, double }], [2 x { double, double }]* %[[VAL_0]], i64 0, i64 1, i32 1
// QIR:         store double 0.000000e+00, double* %[[VAL_5]]
// QIR:         %[[VAL_6:.*]] = call %[[VAL_7:.*]]* @__quantum__rt__qubit_allocate_array_with_state_complex64(i64 1, { double, double }* {{.*}}%[[VAL_1]])
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_7]]* %[[VAL_6]], i64 0)
// QIR:         %[[VAL_10:.*]] = load %[[VAL_9]]*, %[[VAL_9]]** %[[VAL_8]]
// QIR:         %[[VAL_11:.*]] = call %[[VAL_12:.*]]* @__quantum__qis__mz(%[[VAL_9]]* %[[VAL_10]])
// QIR:         %[[VAL_13:.*]] = bitcast %[[VAL_12]]* %[[VAL_11]] to i1*
// QIR:         %[[VAL_14:.*]] = load i1, i1* %[[VAL_13]]
// QIR:         call void @__quantum__rt__qubit_release_array(%[[VAL_7]]* %[[VAL_6]])
// QIR:         ret i1 %[[VAL_14]]
// QIR:       }

