/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck --check-prefixes=CHECK,ALIVE %s
// RUN: cudaq-quake %cpp_std %s | cudaq-opt -erase-noise | FileCheck --check-prefixes=CHECK,DEAD %s
// RUN: cudaq-quake %cpp_std %s | cudaq-opt | cudaq-translate --convert-to=qir | FileCheck --check-prefix=QIR %s
// clang-format on

#include <cudaq.h>

struct SantaKraus : public cudaq::kraus_channel {
  // vaporware here.
};

struct testApplyNoise {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    cudaq::apply_noise<SantaKraus>(q0, q1);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testApplyNoise() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// ALIVE:           quake.apply_noise @_ZN5cudaq11apply_noiseI{{.*}}SantaKraus{{.*}}() %[[VAL_0]], %[[VAL_1]] : !quake.ref, !quake.ref
// DEAD-NOT:        quake.apply_noise @_ZN5cudaq11apply_noiseI{{.*}}SantaKraus{{.*}}() %[[VAL_0]], %[[VAL_1]] : !quake.ref, !quake.ref
// CHECK:           return
// CHECK:         }

// QIR-LABEL: define void @__nvqpp__mlirgen__testApplyNoise() local_unnamed_addr {
// QIR:         %[[VAL_0:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 2)
// QIR:         %[[VAL_2:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 0)
// QIR:         %[[VAL_4:.*]] = load %Qubit*, %Qubit** %[[VAL_2]], align 8
// QIR:         %[[VAL_5:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_0]], i64 1)
// QIR:         %[[VAL_6:.*]] = load %Qubit*, %Qubit** %[[VAL_5]], align 8
// QIR:         tail call void @_ZN5cudaq11apply_noise{{.*}}SantaKraus{{.*}}(%Qubit* %[[VAL_4]], %Qubit* %[[VAL_6]])
// QIR:         tail call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_0]])
// QIR:         ret void
// QIR:       }
