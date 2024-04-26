/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake -D CUDAQ_SIMULATION_SCALAR_FP64 %cpp_std %s | cudaq-opt | FileCheck %s
// RUN: cudaq-quake -D CUDAQ_SIMULATION_SCALAR_FP64 %cpp_std %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
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
// CHECK:           %[[VAL_5:.*]] = quake.init_state %[[VAL_4]], %[[VAL_3]] : (!quake.veq<2>, !cc.ptr<!cc.array<f64 x 4>>) -> !quake.veq<?>
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
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4.000000e-01 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 6.000000e-01 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = complex.create %[[VAL_5]], %[[VAL_6]] : complex<f64>
// CHECK:           %[[VAL_8:.*]] = complex.create %[[VAL_4]], %[[VAL_3]] : complex<f64>
// CHECK:           %[[VAL_9:.*]] = complex.create %[[VAL_6]], %[[VAL_5]] : complex<f64>
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][0] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_7]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_10]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_8]], %[[VAL_12]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_10]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_13]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_14:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_15:.*]] = quake.init_state %[[VAL_14]], %[[VAL_10]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<?>
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
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2.500000e-01 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 7.500000e-01 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = complex.create %[[VAL_5]], %[[VAL_6]] : complex<f64>
// CHECK:           %[[VAL_8:.*]] = complex.create %[[VAL_4]], %[[VAL_3]] : complex<f64>
// CHECK:           %[[VAL_9:.*]] = complex.create %[[VAL_6]], %[[VAL_5]] : complex<f64>
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][0] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_7]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_10]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_8]], %[[VAL_12]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_10]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_13]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_14:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_15:.*]] = quake.init_state %[[VAL_14]], %[[VAL_10]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<?>
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
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f{{[1280]+}}
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2.000000e-01 : f64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 8.000000e-01 : f64
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1.000000e+00 : f{{[1280]+}}
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_9:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = call @_ZNSt8literals16complex_literalsli1iEe(%[[VAL_7]]) : (f{{[1280]+}}) -> complex<f64>
// CHECK:           %[[VAL_11:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_12:.*]] = call @_ZStplIdESt7complexIT_ERKS1_RKS2_(%[[VAL_9]], %[[VAL_11]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_13:.*]] = complex.create %[[VAL_6]], %[[VAL_5]] : complex<f64>
// CHECK:           %[[VAL_14:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_14]] : !cc.ptr<f64>
// CHECK:           %[[VAL_15:.*]] = call @_ZNSt8literals16complex_literalsli1iEe(%[[VAL_3]]) : (f{{[1280]+}}) -> complex<f64>
// CHECK:           %[[VAL_16:.*]] = cc.alloca complex<f64>
// CHECK:           cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_17:.*]] = call @_ZStplIdESt7complexIT_ERKS1_RKS2_(%[[VAL_14]], %[[VAL_16]]) : (!cc.ptr<f64>, !cc.ptr<complex<f64>>) -> complex<f64>
// CHECK:           %[[VAL_18:.*]] = cc.alloca !cc.array<complex<f64> x 4>
// CHECK:           %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_18]][0] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_12]], %[[VAL_19]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_18]][1] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_13]], %[[VAL_20]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_18]][2] : (!cc.ptr<!cc.array<complex<f64> x 4>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_17]], %[[VAL_21]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_22:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_23:.*]] = quake.init_state %[[VAL_22]], %[[VAL_18]] : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f64> x 4>>) -> !quake.veq<?>
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
// CHECK:           %[[VAL_3:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<f64>) -> !cc.ptr<f64>
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
// CHECK:           %[[VAL_3:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<f64>) -> !cc.ptr<f64>
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
// CHECK:           %[[VAL_4:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
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
// CHECK:           %[[VAL_4:.*]] = math.cttz %[[VAL_30]] : i64
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>[%[[VAL_4]] : i64]
// CHECK:           %[[VAL_7:.*]] = quake.init_state %[[VAL_6]], %[[VAL_5]] : (!quake.veq<?>, !cc.ptr<complex<f64>>) -> !quake.veq<?>
// clang-format on

__qpu__ auto Strawberry() {
  cudaq::qubit q = {0., 1.};
  return mz(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_Strawberry._Z10Strawberryv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = complex.create %[[VAL_1]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_3:.*]] = complex.create %[[VAL_0]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<complex<f64> x 2>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_8:.*]] = quake.init_state %[[VAL_7]], %[[VAL_4]] : (!quake.veq<1>, !cc.ptr<!cc.array<complex<f64> x 2>>) -> !quake.veq<1>
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_8]][0] : (!quake.veq<1>) -> !quake.ref
// CHECK:           %[[VAL_10:.*]] = quake.mz %[[VAL_9]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_11:.*]] = quake.discriminate %[[VAL_10]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_11]] : i1
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
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.70710678118654757 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = complex.create %[[VAL_0]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_3:.*]] = complex.create %[[VAL_0]], %[[VAL_1]] : complex<f64>
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<complex<f64> x 2>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<complex<f64> x 2>>) -> !cc.ptr<complex<f64>>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<complex<f64>>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_8:.*]] = quake.init_state %[[VAL_7]], %[[VAL_4]] : (!quake.veq<1>, !cc.ptr<!cc.array<complex<f64> x 2>>) -> !quake.veq<1>
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
// QIR:         %[[VAL_0:.*]] = tail call %[[VAL_1:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 2, i8* nonnull bitcast ([4 x double]* @__nvqpp__rodata_init_0 to i8*))
// QIR:         %[[VAL_2:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%[[VAL_1]]* %[[VAL_0]])
// QIR:       }

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
// QIR:         call void @llvm.memset
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 2, i8* nonnull %[[VAL_7]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:       }

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
// QIR:         call void @llvm.memset
// QIR:         %[[VAL_8:.*]] = call %[[VAL_9:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 2, i8* nonnull %[[VAL_7]])
// QIR:         %[[VAL_10:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_9]]* %[[VAL_8]])
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__RockyRoad() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = alloca double, align 8
// QIR:         store double 0.000000e+00, double* %[[VAL_0]], align 8
// QIR:         %[[VAL_1:.*]] = tail call { double, double } @_ZNSt8literals16complex_literalsli1iEe(
// QIR:         %[[VAL_2:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_3:.*]] = extractvalue { double, double } %[[VAL_1]], 0
// QIR:         %[[VAL_4:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 0
// QIR:         store double %[[VAL_3]], double* %[[VAL_4]], align 8
// QIR:         %[[VAL_5:.*]] = extractvalue { double, double } %[[VAL_1]], 1
// QIR:         %[[VAL_6:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_2]], i64 0, i32 1
// QIR:         store double %[[VAL_5]], double* %[[VAL_6]], align 8
// QIR:         %[[VAL_7:.*]] = call { double, double } @_ZStplIdESt7complexIT_ERKS1_RKS2_(double* nonnull %[[VAL_0]], { double, double }* nonnull %[[VAL_2]])
// QIR:         %[[VAL_8:.*]] = alloca double, align 8
// QIR:         store double 1.000000e+00, double* %[[VAL_8]], align 8
// QIR:         %[[VAL_9:.*]] = call { double, double } @_ZNSt8literals16complex_literalsli1iEe(
// QIR:         %[[VAL_10:.*]] = alloca { double, double }, align 8
// QIR:         %[[VAL_11:.*]] = extractvalue { double, double } %[[VAL_9]], 0
// QIR:         %[[VAL_12:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 0
// QIR:         store double %[[VAL_11]], double* %[[VAL_12]], align 8
// QIR:         %[[VAL_13:.*]] = extractvalue { double, double } %[[VAL_9]], 1
// QIR:         %[[VAL_14:.*]] = getelementptr inbounds { double, double }, { double, double }* %[[VAL_10]], i64 0, i32 1
// QIR:         store double %[[VAL_13]], double* %[[VAL_14]], align 8
// QIR:         %[[VAL_15:.*]] = call { double, double } @_ZStplIdESt7complexIT_ERKS1_RKS2_(double* nonnull %[[VAL_8]], { double, double }* nonnull %[[VAL_10]])
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
// QIR:         %[[VAL_27:.*]] = bitcast [4 x { double, double }]* %[[VAL_16]] to i8*
// QIR:         %[[VAL_28:.*]] = call %[[VAL_29:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 2, i8* nonnull %[[VAL_27]])
// QIR:         %[[VAL_30:.*]] = call i64 @__quantum__rt__array_get_size_1d(%[[VAL_29]]* %[[VAL_28]])
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__Pistachio() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_I:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR: %[[VAL_1:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_I]], i1 false)
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_3:.*]] = bitcast double* %[[VAL_2]] to i8*
// QIR:         %[[VAL_4:.*]] = tail call %[[VAL_5:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_1]], i8* %[[VAL_3]])
// QIR:         %[[VAL_6:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%[[VAL_5]]* %[[VAL_4]])
// QIR:       }

// QIR-LABEL: define i1 @__nvqpp__mlirgen__ChocolateMint() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { double*, i64 } @_Z15getTwoTimesRankv()
// QIR:         %[[VAL_I:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
// QIR: %[[VAL_1:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_I]], i1 false)
// QIR:         %[[VAL_2:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_3:.*]] = bitcast double* %[[VAL_2]] to i8*
// QIR:         %[[VAL_4:.*]] = tail call %[[VAL_5:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_1]], i8* %[[VAL_3]])
// QIR:         %[[VAL_6:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%[[VAL_5]]* %[[VAL_4]])
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__Neapolitan() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_I:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR: %[[VAL_1:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_I]], i1 false)
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_3:.*]] = bitcast { double, double }* %[[VAL_2]] to i8*
// QIR:         %[[VAL_4:.*]] = tail call %[[VAL_5:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_1]], i8* %[[VAL_3]])
// QIR:         %[[VAL_6:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%[[VAL_5]]* %[[VAL_4]])
// QIR:       }

// QIR-LABEL: define { i1*, i64 } @__nvqpp__mlirgen__ButterPecan() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call { { double, double }*, i64 } @_Z14getComplexInitv()
// QIR:         %[[VAL_I:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 1
// QIR: %[[VAL_1:.*]] = tail call i64 @llvm.cttz.i64(i64 %[[VAL_I]], i1 false)
// QIR:         %[[VAL_2:.*]] = extractvalue { { double, double }*, i64 } %[[VAL_0]], 0
// QIR:         %[[VAL_3:.*]] = bitcast { double, double }* %[[VAL_2]] to i8*
// QIR:         %[[VAL_4:.*]] = tail call %[[VAL_5:.*]]* @__quantum__rt__qubit_allocate_array_with_state_fp64(i64 %[[VAL_1]], i8* %[[VAL_3]])
// QIR:         %[[VAL_6:.*]] = tail call i64 @__quantum__rt__array_get_size_1d(%[[VAL_5]]* %[[VAL_4]])
// QIR:       }

