/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void qview_test(cudaq::qview<> v) {}

__qpu__ void qvector_test(cudaq::qvector<> v) {}

__qpu__ void qarray_test(cudaq::qarray<4> a) {}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qview_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>)

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qvector_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>)

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qarray_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<4>)


