/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize --lower-to-cfg | cudaq-translate --convert-to=qir -o - | FileCheck %s

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

// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK:         tail call %{{.*}}* @__quantum__rt__qubit_allocate_array(i64 3)
// CHECK:         tail call i8* @__quantum__rt__array_get_element_ptr_1d(%{{.*}}* %{{.*}}, i64 1)
// CHECK:         tail call void @__quantum__qis__h(%{{.*}}* %{{.*}})
// CHECK:         tail call i8* @__quantum__rt__array_get_element_ptr_1d(%{{.*}}* %{{.*}}, i64 2)
// CHECK:         tail call void (i64, void (%{{.*}}*, %{{.*}}*)*, ...) @invokeWithControlQubits(i64 1, void (%{{.*}}*, %{{.*}}*)* nonnull @__quantum__qis__x__ctl, %{{.*}}* %{{.*}}, %{{.*}}* %{{.*}})
// CHECK:         tail call i8* @__quantum__rt__array_get_element_ptr_1d(%{{.*}}* %{{.*}}, i64 0)
// CHECK:         tail call void (i64, void (%{{.*}}*, %{{.*}}*)*, ...) @invokeWithControlQubits(i64 1, void (%{{.*}}*, %{{.*}}*)* nonnull @__quantum__qis__x__ctl, %{{.*}}* %{{.*}}, %{{.*}}* %{{.*}})
// CHECK:         tail call void @__quantum__qis__h(%{{.*}}* %{{.*}})
// CHECK:         tail call %{{.*}}* @__quantum__qis__mz__to__register(%{{.*}}* %{{.*}}, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623000, i64 0, i64 0))
// CHECK:         tail call %{{.*}}* @__quantum__qis__mz__to__register(%{{.*}}* %{{.*}}, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.623100, i64 0, i64 0))
// CHECK:         tail call void @__quantum__rt__qubit_release_array(%{{.*}}* %{{.*}})
// CHECK:         ret void
// CHECK:       }

