/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ExecutionContext.h"
#include <cstdint>

// Define some stubs for the QIR opaque types
class Array;
class Qubit;
class Result;
using TuplePtr = int8_t *;

/// QIR QIS external declarations
extern "C" {
void __quantum__rt__deallocate_all(const std::size_t, const std::size_t *);
Array *__quantum__rt__array_concatenate(Array *, Array *);
void __quantum__rt__array_release(Array *);
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t);
int64_t __quantum__rt__array_get_size_1d(Array *);
Array *__quantum__rt__array_create_1d(int, int64_t);
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits);
void __quantum__qis__measure__body(Array *, Array *);
Qubit *__quantum__rt__qubit_allocate();
Array *__quantum__rt__qubit_allocate_array(uint64_t);
void __quantum__rt__qubit_release(Qubit *);
void __quantum__rt__setExecutionContext(cudaq::ExecutionContext *);
void __quantum__rt__resetExecutionContext();

void __quantum__qis__reset(Qubit *);

void __quantum__qis__h(Qubit *q);
void __quantum__qis__h__ctl(Array *ctls, Qubit *q);

void __quantum__qis__x(Qubit *q);
void __quantum__qis__x__ctl(Array *ctls, Qubit *q);

void __quantum__qis__y(Qubit *q);
void __quantum__qis__y__ctl(Array *ctls, Qubit *q);

void __quantum__qis__z(Qubit *q);
void __quantum__qis__z__ctl(Array *ctls, Qubit *q);

void __quantum__qis__t(Qubit *q);
void __quantum__qis__t__ctl(Array *ctls, Qubit *q);
void __quantum__qis__tdg(Qubit *q);
void __quantum__qis__tdg__ctl(Array *ctls, Qubit *q);

void __quantum__qis__s(Qubit *q);
void __quantum__qis__s__ctl(Array *ctls, Qubit *q);
void __quantum__qis__sdg(Qubit *q);
void __quantum__qis__sdg__ctl(Array *ctls, Qubit *q);

void __quantum__qis__rx(double, Qubit *q);
void __quantum__qis__rx__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__ry(double, Qubit *q);
void __quantum__qis__ry__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__rz(double, Qubit *q);
void __quantum__qis__rz__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__r1(double, Qubit *q);
void __quantum__qis__r1__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__swap(Qubit *, Qubit *);
void __quantum__qis__swap__ctl(Array *, Qubit *, Qubit *);
void __quantum__qis__cphase(double x, Qubit *src, Qubit *tgt);

Result *__quantum__qis__mz(Qubit *);
}
