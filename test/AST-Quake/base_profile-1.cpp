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
// BASE:         tail call void @__quantum__qis__h__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__x__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__y__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__z__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__t__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__s__body(%Qubit* null)
// BASE:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__s__adj(%Qubit* null)
// BASE:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, %Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__rx__body(double -5.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, %Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// BASE:         tail call void @__quantum__qis__ry__body(double -6.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__rz__body(double 7.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__rz__body(double 0x4021397F62B6AE7E, %Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// BASE:         tail call void @__quantum__qis__rz__body(double 0xC025397F62B6AE7E, %Qubit* null)
// BASE:         tail call void @__quantum__qis__r1__body(double 4.612300e+00, %Qubit* null)
// BASE:         tail call void @__quantum__qis__r1__body(double 0x400CE5FD8ADAB9F6, %Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__r1__body(double 0xBFF9CBFB15B573EC, %Qubit* null)
// BASE:         tail call void @__quantum__qis__swap__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// BASE:         tail call void @__quantum__qis__u3__body(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, %Qubit* nonnull inttoptr (i64 3 to %Qubit*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 2 to %Result*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*), %Result* nonnull inttoptr (i64 3 to %Result*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*), %Result* nonnull inttoptr (i64 4 to %Result*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*), %Result* nonnull inttoptr (i64 5 to %Result*))
// BASE:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*), %Result* nonnull inttoptr (i64 6 to %Result*))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([10 x i8], [10 x i8]* @cstr.73696E676C65746F6E00, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.65696E7300, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*), i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 3 to %Result*), i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 4 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 5 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// BASE:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 6 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// BASE:         ret void
// BASE:       }

// ADAPT-LABEL: define void @__nvqpp__mlirgen__comprehensive()
// ADAPT:         tail call void @__quantum__qis__h__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__h__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__x__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__x__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__y__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__y__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__z__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__z__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__t__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__t__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__t__adj(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__s__body(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__s__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__s__adj(%Qubit* null)
// ADAPT:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__rx__body(double 5.612300e+00, %Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__rx__body(double -5.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__ry__body(double 6.612300e+00, %Qubit* nonnull inttoptr (i64 4 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__ry__body(double -6.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__rz__body(double 7.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__rz__body(double 0x4021397F62B6AE7E, %Qubit* nonnull inttoptr (i64 5 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__rz__body(double 0xC025397F62B6AE7E, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__r1__body(double 4.612300e+00, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__r1__body(double 0x400CE5FD8ADAB9F6, %Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__r1__body(double 0xBFF9CBFB15B573EC, %Qubit* null)
// ADAPT:         tail call void @__quantum__qis__swap__body(%Qubit* null, %Qubit* nonnull inttoptr (i64 6 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__u3__body(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, %Qubit* nonnull inttoptr (i64 3 to %Qubit*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([10 x i8], [10 x i8]* @cstr.73696E676C65746F6E00, i64 0, i64 0))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 1 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.65696E7300, i64 0, i64 0))
// ADAPT:         %[[VAL_2:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 2 to %Qubit*), %Result* nonnull inttoptr (i64 2 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 2 to %Result*), i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// ADAPT:         %[[VAL_3:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 2 to %Result*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 3 to %Qubit*), %Result* nonnull inttoptr (i64 3 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 3 to %Result*), i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// ADAPT:         %[[VAL_4:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 3 to %Result*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 4 to %Qubit*), %Result* nonnull inttoptr (i64 4 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 4 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// ADAPT:         %[[VAL_5:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 4 to %Result*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 5 to %Qubit*), %Result* nonnull inttoptr (i64 5 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 5 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// ADAPT:         %[[VAL_6:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 5 to %Result*))
// ADAPT:         tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 6 to %Qubit*), %Result* nonnull inttoptr (i64 6 to %Result*))
// ADAPT:         tail call void @__quantum__rt__result_record_output(%Result* nonnull inttoptr (i64 6 to %Result*), i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// ADAPT:         %[[VAL_7:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 6 to %Result*))
// ADAPT:         ret void
// ADAPT:       }

// FULL-LABEL: define void @__nvqpp__mlirgen__comprehensive()
// FULL:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 9)
// FULL:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// FULL:         %[[VAL_4:.*]] = bitcast %Qubit** %[[VAL_2]] to i8**
// FULL:         %[[VAL_5:.*]] = load i8*, i8** %[[VAL_4]], align 8
// FULL:         %[[VAL_6:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// FULL:         %[[VAL_7:.*]] = bitcast %Qubit** %[[VAL_6]] to i8**
// FULL:         %[[VAL_8:.*]] = load i8*, i8** %[[VAL_7]], align 8
// FULL:         %[[VAL_9:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 2)
// FULL:         %[[VAL_10:.*]] = load %Qubit*, %Qubit** %[[VAL_9]], align 8
// FULL:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_10]])
// FULL:         %[[VAL_11:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 6)
// FULL:         %[[VAL_12:.*]] = load %Qubit*, %Qubit** %[[VAL_11]], align 8
// FULL:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_12]])
// FULL:         %[[VAL_13:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 7)
// FULL:         %[[VAL_14:.*]] = load %Qubit*, %Qubit** %[[VAL_13]], align 8
// FULL:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_14]])
// FULL:         %[[VAL_15:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 8)
// FULL:         %[[VAL_16:.*]] = load %Qubit*, %Qubit** %[[VAL_15]], align 8
// FULL:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_16]])
// FULL:         %[[VAL_17:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 4)
// FULL:         %[[VAL_18:.*]] = load %Qubit*, %Qubit** %[[VAL_17]], align 8
// FULL:         %[[VAL_19:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 5)
// FULL:         %[[VAL_20:.*]] = load %Qubit*, %Qubit** %[[VAL_19]], align 8
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_18]], %Qubit* %[[VAL_20]], i8* %[[VAL_8]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__h__ctl to i8*), i8* %[[VAL_8]], %Qubit* %[[VAL_10]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_18]], %Qubit* %[[VAL_20]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_16]])
// FULL:         %[[VAL_21:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 3)
// FULL:         %[[VAL_22:.*]] = load %Qubit*, %Qubit** %[[VAL_21]], align 8
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_18]], %Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__y(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__y(%Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__y(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__y(%Qubit* %[[VAL_16]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_12]], %Qubit* %[[VAL_14]], i8* %[[VAL_8]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_16]], i8* %[[VAL_8]], i8* %[[VAL_5]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__y__ctl to i8*), i8* %[[VAL_5]], %Qubit* %[[VAL_22]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_16]], i8* %[[VAL_8]], i8* %[[VAL_5]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_12]], %Qubit* %[[VAL_14]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__z(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__z(%Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__z(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__z(%Qubit* %[[VAL_16]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__z__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__z__ctl to i8*), %Qubit* %[[VAL_18]], %Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__t(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__t(%Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__t(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__t(%Qubit* %[[VAL_16]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__t__ctl to i8*), i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void @__quantum__qis__t__adj(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_12]])
// FULL:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__s(%Qubit* %[[VAL_16]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__s__ctl to i8*), i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void @__quantum__qis__s__adj(%Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__rx(double 5.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__rx(double 5.612300e+00, %Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (double, %Array*, %Qubit*)* @__quantum__qis__rx__ctl to i8*), double 5.612300e+00, i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void @__quantum__qis__rx(double -5.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__ry(double 6.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__ry(double 6.612300e+00, %Qubit* %[[VAL_12]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_12]], i8* %[[VAL_8]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (double, %Array*, %Qubit*)* @__quantum__qis__ry__ctl to i8*), double 6.612300e+00, i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_12]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__ry(double -6.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__rz(double 7.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__rz(double 0x4021397F62B6AE7E, %Qubit* %[[VAL_14]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (double, %Array*, %Qubit*)* @__quantum__qis__rz__ctl to i8*), double 0x4023397F62B6AE7E, i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_20]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__x(%Qubit* %[[VAL_20]])
// FULL:         tail call void @__quantum__qis__rz(double 0xC025397F62B6AE7E, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__r1(double 4.612300e+00, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__r1(double 0x400CE5FD8ADAB9F6, %Qubit* %[[VAL_16]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_14]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 1, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (double, %Array*, %Qubit*)* @__quantum__qis__r1__ctl to i8*), double 0x4004E5FD8ADAB9F6, i8* %[[VAL_8]], %Qubit* %[[VAL_22]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_14]], %Qubit* %[[VAL_16]], i8* %[[VAL_8]])
// FULL:         tail call void @__quantum__qis__r1(double 0xBFF9CBFB15B573EC, %Qubit* %[[VAL_10]])
// FULL:         tail call void @__quantum__qis__swap(%Qubit* %[[VAL_10]], %Qubit* %[[VAL_16]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 1, i64 2, i8* nonnull bitcast (void (%Array*, %Qubit*, %Qubit*)* @__quantum__qis__swap__ctl to i8*), %Qubit* %[[VAL_10]], %Qubit* %[[VAL_22]], %Qubit* %[[VAL_20]])
// FULL:         tail call void @__quantum__qis__u3(double 8.000000e-01, double 5.000000e-01, double -1.000000e+00, %Qubit* %[[VAL_20]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_14]], %Qubit* %[[VAL_12]], i8* %[[VAL_8]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 3, i64 0, i64 1, i64 1, i8* nonnull bitcast (void (double, double, double, %Array*, %Qubit*)* @__quantum__qis__u3__ctl to i8*), double 6.200000e+00, double -3.100000e+00, double 0x401F333333333333, i8* %[[VAL_8]], %Qubit* %[[VAL_10]])
// FULL:         tail call void (i64, i64, i64, i64, i8*, ...) @generalizedInvokeWithRotationsControlsTargets(i64 0, i64 0, i64 2, i64 1, i8* nonnull bitcast (void (%Array*, %Qubit*)* @__quantum__qis__x__ctl to i8*), %Qubit* %[[VAL_14]], %Qubit* %[[VAL_12]], i8* %[[VAL_8]])
// FULL:         %[[VAL_23:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_10]], i8* nonnull getelementptr inbounds ([10 x i8], [10 x i8]* @cstr.73696E676C65746F6E00, i64 0, i64 0))
// FULL:         %[[VAL_25:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_22]], i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.65696E7300, i64 0, i64 0))
// FULL:         %[[VAL_26:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_18]], i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// FULL:         %[[VAL_27:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_20]], i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.64756200, i64 0, i64 0))
// FULL:         %[[VAL_28:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_12]], i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// FULL:         %[[VAL_29:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_14]], i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// FULL:         %[[VAL_30:.*]] = tail call %Result* @__quantum__qis__mz__to__register(%Qubit* %[[VAL_16]], i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @cstr.7472697000, i64 0, i64 0))
// FULL:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// FULL:         ret void
// FULL:       }
// FULL:         ret void

