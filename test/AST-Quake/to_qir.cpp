/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt --lower-to-cfg | cudaq-translate --convert-to=qir -o - | FileCheck %s

#include <cudaq.h>

struct kernel {
    void operator()() __qpu__ {
        cudaq::qarray<3> q;
        h(q[1]);
        x<cudaq::ctrl>(q[1], q[2]);

        x<cudaq::ctrl>(q[0], q[1]);
        h(q[0]);

        auto b0 = mz(q[0]);
        auto b1 = mz(q[1]);

        if (b1) x(q[2]);
        if (b0) z(q[2]);
    }
};

// CHECK-LABEL: define void @__nvqpp__mlirgen__kernel()
// CHECK:         tail call target("qir#Array") @__quantum__rt__qubit_allocate_array(i64 3)
// CHECK:         tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %{{.*}}, i64 1)
// CHECK:         tail call void @__quantum__qis__h(target("qir#Qubit") %{{.*}})
// CHECK:         tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %{{.*}}, i64 2)
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 1, ptr nonnull @__quantum__qis__x__ctl, target("qir#Qubit") %{{.*}}, target("qir#Qubit") %{{.*}})
// CHECK:         tail call target("qir#Qubit") @__quantum__rt__array_get_qubit_element(target("qir#Array") %{{.*}}, i64 0)
// CHECK:         tail call void (i64, ptr, ...) @invokeWithControlQubits(i64 1, ptr nonnull @__quantum__qis__x__ctl, target("qir#Qubit") %{{.*}}, target("qir#Qubit") %{{.*}})
// CHECK:         tail call void @__quantum__qis__h(target("qir#Qubit") %{{.*}})
// CHECK:         tail call target("qir#Result") @__quantum__qis__mz__to__register(target("qir#Qubit") %{{.*}}, ptr nonnull @cstr.{{.*}})
// CHECK:         tail call target("qir#Result") @__quantum__qis__mz__to__register(target("qir#Qubit") %{{.*}}, ptr nonnull @cstr.{{.*}})
// CHECK:         tail call void @__quantum__rt__qubit_release_array(target("qir#Array") %{{.*}})
// CHECK:         ret void
// CHECK:       }

