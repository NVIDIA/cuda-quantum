/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake -D DAGGER %s | FileCheck --check-prefixes=DAGGER %s

#include <cudaq.h>

#ifdef DAGGER
#define CALL compute_dag_action
#else
#define CALL compute_action
#endif

void t() __qpu__ {
	cudaq::qreg r(5);
	cudaq:: CALL (
		[&](){ t(r[0]); x(r[1]); },
		[&](){ h(r[2]); });
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_t
// CHECK-SAME: () attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
// CHECK:           %[[VAL_3:.*]] = cc.create_lambda {
// CHECK:               quake.t %{{.*}}
// CHECK:               quake.x %{{.*}}
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = cc.create_lambda {
// CHECK:               quake.h %{{.*}}
// CHECK:           }
// CHECK:           quake.compute_action %[[VAL_3]], %[[VAL_10]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
// CHECK:           return
// CHECK:         }

// DAGGER-LABEL:   func.func @__nvqpp__mlirgen__function_t
// DAGGER-SAME: () attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
// DAGGER:           %[[VAL_3:.*]] = cc.create_lambda {
// DAGGER:               quake.t %{{.*}}
// DAGGER:               quake.x %{{.*}}
// DAGGER:           }
// DAGGER:           %[[VAL_10:.*]] = cc.create_lambda {
// DAGGER:               quake.h %{{.*}}
// DAGGER:           }
// DAGGER:           quake.compute_action<dag> %[[VAL_3]], %[[VAL_10]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
// DAGGER:           return
// DAGGER:         }

