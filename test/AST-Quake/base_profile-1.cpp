/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --decomposition=enable-patterns="CCXToCCZ,CCZToCX" --combine-quantum-alloc --canonicalize | cudaq-translate --convert-to=qir-base | FileCheck --check-prefix=BASE %s
// RUN: cudaq-quake %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --decomposition=enable-patterns="CCXToCCZ,CCZToCX" --combine-quantum-alloc --canonicalize | cudaq-translate --convert-to=qir-adaptive | FileCheck --check-prefix=ADAPT %s
// RUN: cudaq-quake -D FULL %s | cudaq-opt --add-dealloc --expand-measurements --factor-quantum-alloc --expand-control-veqs --cc-loop-unroll --canonicalize --multicontrol-decomposition --lower-to-cfg --cse --combine-quantum-alloc --canonicalize | cudaq-translate --convert-to=qir | FileCheck --check-prefix=FULL %s
// clang-format on

#include <cudaq.h>

#ifdef FULL
#define T(X) X
#else
#define T(X)
#endif

// This test is meant to exercise more of the NVQIR calls. The ctrl variants are
// commented out because they don't work with qir-base or qir-adaptive.
struct comprehensive {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qarray<1> q1;
    cudaq::qarray<2> q2;
    cudaq::qarray<3> q3;

    // The basic one target gates.
    h(q0);
    h(q3);
    T(h<cudaq::ctrl>(q2[0], q2[1], q0));
    x(q0);
    x(q3);
    T(x<cudaq::ctrl>(!q2[1], q3[2], q1[0]));
    cnot(q2[0], q3[0]);
    y(q0);
    y(q3);
    T(y<cudaq::ctrl>(q3, q1[0]));
    z(q0);
    z(q3);
    T(z<cudaq::ctrl>(!q2[1], q3[2], q1[0]));
    cz(q2[0], q3[0]);
    t(q0);
    t(q3);
    T(t<cudaq::ctrl>(!q2[1], q3[2], q1[0]));
    t<cudaq::adj>(q3[1]);
    s(q0);
    s(q3);
    T(s<cudaq::ctrl>(!q2[1], q3[2], q1[0]));
    s<cudaq::adj>(q0);

    // Basic rotations.
    const double p = 5.6123;
    rx(p, q0);
    rx(p, q3[1]);
    T(rx<cudaq::ctrl>(p, !q2[1], q3[2], q1[0]));
    rx<cudaq::adj>(p, q0);

    ry(p + 1, q0);
    ry(p + 1, q3[0]);
    T(ry<cudaq::ctrl>(p + 1, q2[1], q3[0], q1[0]));
    ry<cudaq::adj>(p + 1, q0);

    rz(p + 2, q0);
    rz(p + 3, q3[1]);
    T(rz<cudaq::ctrl>(p + 4, !q2[1], q3[2], q1[0]));
    rz<cudaq::adj>(p + 5, q0);

    r1(p - 1, q0);
    r1(p - 2, q3[2]);
    T(r1<cudaq::ctrl>(p - 3, q3[1], q3[2], q1[0]));
    r1<cudaq::adj>(p - 4, q0);

    // Misc. gates.
    swap(q0, q3[2]);
    T(swap<cudaq::ctrl>(q0, q1[0], q2[1]));

    u3(0.8, 0.5, -1.0, q2[1]);
    T(u3<cudaq::ctrl>(6.2, -3.1, 7.8, q3[1], q3[0], q0));

    auto singleton = mz(q0);
    auto eins = mz(q1);
    auto dub = mz(q2);
    auto trip = mz(q3);
  }
};

// clang-format off

// BASE-LABEL: define void @__nvqpp__mlirgen__comprehensive()
// BASE:         tail call void @__quantum__qis__h__body(ptr null)
// BASE:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__x__body(ptr null)
// BASE:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__y__body(ptr null)
// BASE:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__z__body(ptr null)
// BASE:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__cz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__t__body(ptr null)
// BASE:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__t__adj(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__s__body(ptr null)
// BASE:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__s__adj(ptr null)
// BASE:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__rx__body(double -5.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__ry__body(double -6.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__rz__body(double 7.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__rz__body(double 0x4021397F62B6AE7E, ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__rz__body(double 0xC025397F62B6AE7E, ptr null)
// BASE:         tail call void @__quantum__qis__r1__body(double 4.612300e+00, ptr null)
// BASE:         tail call void @__quantum__qis__r1__body(double 0x400CE5FD8ADAB9F6, ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__r1__body(double 0xBFF9CBFB15B573EC, ptr null)
// BASE:         tail call void @__quantum__qis__swap__body(ptr null, ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__qis__u3__body(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, ptr nonnull inttoptr (i64 3 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr null, ptr null)
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 1 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 3 to ptr), ptr nonnull inttoptr (i64 3 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 4 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 5 to ptr), ptr nonnull inttoptr (i64 5 to ptr))
// BASE:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 6 to ptr), ptr nonnull inttoptr (i64 6 to ptr))
// BASE:         tail call void @__quantum__rt__result_record_output(ptr null, ptr nonnull @cstr.73696E676C65746F6E00)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.65696E7300)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull @cstr.64756200)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 3 to ptr), ptr nonnull @cstr.64756200)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 4 to ptr), ptr nonnull @cstr.7472697000)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 5 to ptr), ptr nonnull @cstr.7472697000)
// BASE:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 6 to ptr), ptr nonnull @cstr.7472697000)
// BASE:         ret void
// BASE:       }

// ADAPT-LABEL: define void @__nvqpp__mlirgen__comprehensive()
// ADAPT:         tail call void @__quantum__qis__h__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__h__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__x__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__x__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__cnot__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__y__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__y__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__z__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__z__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__cz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__t__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__t__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__t__adj(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__s__body(ptr null)
// ADAPT:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__s__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__s__adj(ptr null)
// ADAPT:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__rx__body(double -5.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__ry__body(double -6.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__rz__body(double 7.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__rz__body(double 0x4021397F62B6AE7E, ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__rz__body(double 0xC025397F62B6AE7E, ptr null)
// ADAPT:         tail call void @__quantum__qis__r1__body(double 4.612300e+00, ptr null)
// ADAPT:         tail call void @__quantum__qis__r1__body(double 0x400CE5FD8ADAB9F6, ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__r1__body(double 0xBFF9CBFB15B573EC, ptr null)
// ADAPT:         tail call void @__quantum__qis__swap__body(ptr null, ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__qis__u3__body(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, ptr nonnull inttoptr (i64 3 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr null, ptr null)
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr null, ptr nonnull @cstr.73696E676C65746F6E00)
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull inttoptr (i64 1 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 1 to ptr), ptr nonnull @cstr.65696E7300)
// ADAPT:         %[[VAL_2:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 1 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull inttoptr (i64 2 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 2 to ptr), ptr nonnull @cstr.64756200)
// ADAPT:         %[[VAL_3:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 2 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 3 to ptr), ptr nonnull inttoptr (i64 3 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 3 to ptr), ptr nonnull @cstr.64756200)
// ADAPT:         %[[VAL_4:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 3 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 4 to ptr), ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 4 to ptr), ptr nonnull @cstr.7472697000)
// ADAPT:         %[[VAL_5:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 4 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 5 to ptr), ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 5 to ptr), ptr nonnull @cstr.7472697000)
// ADAPT:         %[[VAL_6:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 5 to ptr))
// ADAPT:         tail call void @__quantum__qis__mz__body(ptr nonnull inttoptr (i64 6 to ptr), ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         tail call void @__quantum__rt__result_record_output(ptr nonnull inttoptr (i64 6 to ptr), ptr nonnull @cstr.7472697000)
// ADAPT:         %[[VAL_7:.*]] = tail call i1 @__quantum__qis__read_result__body(ptr nonnull inttoptr (i64 6 to ptr))
// ADAPT:         ret void
// ADAPT:       }

// FULL-LABEL: define void @__nvqpp__mlirgen__comprehensive()
// FULL:         %[[VAL_0:.*]] = tail call ptr @__quantum__rt__qubit_allocate_array(i64 9)
// FULL:         %[[VAL_2:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 0)
// FULL:         %[[Q0:.*]] = load ptr, ptr %[[VAL_2]], align 8
// FULL:         %[[VAL_4:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 1)
// FULL:         %[[Q1:.*]] = load ptr, ptr %[[VAL_4]], align 8
// FULL:         %[[VAL_6:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 2)
// FULL:         %[[Q2:.*]] = load ptr, ptr %[[VAL_6]], align 8
// FULL:         %[[VAL_8:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 3)
// FULL:         %[[Q3:.*]] = load ptr, ptr %[[VAL_8]], align 8
// FULL:         %[[VAL_10:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 4)
// FULL:         %[[Q4:.*]] = load ptr, ptr %[[VAL_10]], align 8
// FULL:         %[[VAL_12:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 5)
// FULL:         %[[Q5:.*]] = load ptr, ptr %[[VAL_12]], align 8
// FULL:         %[[VAL_14:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 6)
// FULL:         %[[Q6:.*]] = load ptr, ptr %[[VAL_14]], align 8
// FULL:         %[[VAL_16:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 7)
// FULL:         %[[Q7:.*]] = load ptr, ptr %[[VAL_16]], align 8
// FULL:         %[[VAL_18:.*]] = tail call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_0]], i64 8)
// FULL:         %[[Q8:.*]] = load ptr, ptr %[[VAL_18]], align 8
// FULL:         tail call void @__quantum__qis__h(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__h(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__h(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__h(ptr %[[Q8]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q4]], ptr %[[Q5]], ptr %[[Q1]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__h__ctl, ptr %[[Q1]], ptr %[[Q2]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q4]], ptr %[[Q5]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q8]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q4]], ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__y(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__y(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__y(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__y(ptr %[[Q8]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q6]], ptr %[[Q7]], ptr %[[Q1]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q8]], ptr %[[Q1]], ptr %[[Q0]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__y__ctl, ptr %[[Q0]], ptr %[[Q3]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q8]], ptr %[[Q1]], ptr %[[Q0]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q6]], ptr %[[Q7]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__z(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__z(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__z(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__z(ptr %[[Q8]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__z__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__z__ctl, ptr %[[Q4]], ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__t(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__t(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__t(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__t(ptr %[[Q8]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__t__ctl, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void @__quantum__qis__t__adj(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__s(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__s(ptr %[[Q6]])
// FULL:         tail call void @__quantum__qis__s(ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__s(ptr %[[Q8]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__s__ctl, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void @__quantum__qis__s__adj(ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__rx(double 5.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__rx(double 5.612300e+00, ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__rx__ctl, double 5.612300e+00, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void @__quantum__qis__rx(double -5.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__ry(double 6.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__ry(double 6.612300e+00, ptr %[[Q6]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q6]], ptr %[[Q1]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__ry__ctl, double 6.612300e+00, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q6]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__ry(double -6.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__rz(double 7.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__rz(double 0x4021397F62B6AE7E, ptr %[[Q7]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__rz__ctl, double 0x4023397F62B6AE7E, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q5]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__x(ptr %[[Q5]])
// FULL:         tail call void @__quantum__qis__rz(double 0xC025397F62B6AE7E, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__r1(double 4.612300e+00, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__r1(double 0x400CE5FD8ADAB9F6, ptr %[[Q8]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q7]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__r1__ctl, double 0x4004E5FD8ADAB9F6, ptr %[[Q1]], ptr %[[Q3]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q7]], ptr %[[Q8]], ptr %[[Q1]])
// FULL:         tail call void @__quantum__qis__r1(double 0xBFF9CBFB15B573EC, ptr %[[Q2]])
// FULL:         tail call void @__quantum__qis__swap(ptr %[[Q2]], ptr %[[Q8]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 2, ptr nonnull @__quantum__qis__swap__ctl, ptr %[[Q2]], ptr %[[Q3]], ptr %[[Q5]])
// FULL:         tail call void @__quantum__qis__u3(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, ptr %[[Q5]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q7]], ptr %[[Q6]], ptr %[[Q1]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 3, i64 0, i64 1, i64 1, ptr nonnull @__quantum__qis__u3__ctl, double 6.200000e+00, double -3.100000e+00, double 0x401F333333333333, ptr %[[Q1]], ptr %[[Q2]])
// FULL:         tail call void (i64, i64, i64, i64, ptr, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, ptr nonnull @__quantum__qis__x__ctl, ptr %[[Q7]], ptr %[[Q6]], ptr %[[Q1]])
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q2]], ptr nonnull @cstr.73696E676C65746F6E00)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q3]], ptr nonnull @cstr.65696E7300)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q4]], ptr nonnull @cstr.64756200)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q5]], ptr nonnull @cstr.64756200)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q6]], ptr nonnull @cstr.7472697000)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q7]], ptr nonnull @cstr.7472697000)
// FULL:         %{{.*}} = tail call ptr @__quantum__qis__mz__to__register(ptr %[[Q8]], ptr nonnull @cstr.7472697000)
// FULL:         tail call void @__quantum__rt__qubit_release_array(ptr %[[VAL_0]])
// FULL:         ret void
// FULL:       }
// FULL:         ret void

