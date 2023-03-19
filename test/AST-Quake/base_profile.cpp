/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --lower-to-cfg | cudaq-translate --convert-to=qir-base -o - | FileCheck %s

#include <cudaq.h>

struct kernel {
    void operator()() __qpu__ {
        cudaq::qreg<3> q;
        h(q[1]);
        x<cudaq::ctrl>(q[1],q[2]);

        x<cudaq::ctrl>(q[0], q[1]);
        h(q[0]);

        auto b0 = mz(q[0]);
        auto b1 = mz(q[1]);

        if (b1) x(q[2]);
        if (b0) z(q[2]);
    }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel
// CHECK-SAME: ()
// CHECK: tail call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
// CHECK: %[[VAL_2:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* null)
// CHECK: tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* nonnull inttoptr (i64 1 to %Result*))
// CHECK: %[[VAL_3:.*]] = tail call i1 @__quantum__qis__read_result__body(%Result* nonnull inttoptr (i64 1 to %Result*))
// CHECK: br i1 %[[VAL_3]], label %[[VAL_4:.*]], label %[[VAL_5:.*]]
// CHECK: br i1 %[[VAL_2]], label %[[VAL_7:.*]], label %[[VAL_8:.*]]
