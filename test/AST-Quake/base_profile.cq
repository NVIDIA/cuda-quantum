/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --lower-to-cfg | cudaq-translate --convert-to=qir-base -o - | FileCheck %s

#include <cudaq.h>

struct kernel {
    void operator()() __qpu__ {
        cudaq::qreg<3> q;
        h(q[1]);
        x<cudaq::ctrl>(q[1],q[2]);

        x<cudaq::ctrl>(q[0], q[1]);
        h(q[0]);

        // This scope block is intentionally blank and is used for robustness testing.
        {}

        auto b0 = mz(q[0]);
        auto b1 = mz(q[1]);
    }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK:         tail call void @__quantum__qis__mz__body(%{{.*}}* null, %{{.*}}* null)
// CHECK:         tail call void @__quantum__qis__mz__body(%{{.*}}* nonnull inttoptr (i64 1 to %{{.*}}*), %{{.*}}* nonnull inttoptr (i64 1 to %{{.*}}*))
// CHECK:         tail call void @__quantum__rt__result_record_output(%{{.*}}* null, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623000, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__result_record_output(%{{.*}}* nonnull inttoptr (i64 1 to %{{.*}}*), i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623100, i64 0, i64 0))


