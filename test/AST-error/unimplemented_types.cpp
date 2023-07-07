/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake -verify %s

#include <cudaq.h>

// expected-error@*{{type is not yet implemented}}
__qpu__ void qview_test(cudaq::qview<> v) {} // expected-error{{kernel argument type not supported}}

// expected-error@*{{type is not yet implemented}}
__qpu__ void qvector_test(cudaq::qvector<> v) {} // expected-error{{kernel argument type not supported}}

// expected-error@*{{type is not yet implemented}}
__qpu__ void qarray_test(cudaq::qarray<4> a) {} // expected-error{{kernel argument type not supported}}
