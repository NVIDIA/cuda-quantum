// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s -split-input-file -verify-diagnostics

%e = qtx.alloca : !qtx.wire_array<5, dead = 1> // expected-error {{must not return a wire array with dead wires}}

// -----

// expected-error@+1 {{must be provided at least one wire as input}}
%e = qtx.array_create : !qtx.wire_array<0>

// -----

%w0 = qtx.alloca : !qtx.wire
%w1 = qtx.alloca : !qtx.wire
// expected-error@+1 {{There is a mismatch between the number of wires provided and the size of the returned array}}
%e = qtx.array_create %w0, %w1 : !qtx.wire_array<3>

// -----

%array = qtx.alloca : !qtx.wire_array<2>
%i = arith.constant 0 : index
// expected-error@+1 {{must return an array with same size of the input array}}
%q0, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire_array<1, dead = 1>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
// expected-error@+1 {{must return an array with 1 dead wire(s)}}
%q0, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
// expected-error@+1 {{must return a list with 1 wire(s) instead of 2}}
%q0, %q1, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<1, dead = 2>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
%q0, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>
// expected-error@+1 {{cannot borrow from an array of dead wire(s)}}
%q1, %e = qtx.array_borrow %i from %new_array : index from !qtx.wire_array<1, dead = 1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
%j = arith.constant 1 : index
// expected-error@+1 {{cannot borrow 2 wire(s) from an array that only has 1 wire(s) alive}}
%q0, %q1, %new_array = qtx.array_borrow %i, %j from %array : index, index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire, !qtx.wire_array<1, dead = 2>

// -----
%array = qtx.alloca : !qtx.wire_array<1>
%q0 = qtx.alloca : !qtx.wire
// expected-error@+1 {{cannot yield a wire back to an array of alive wire(s)}}
%new_array = qtx.array_yield %q0 to %array : !qtx.wire_array<1> -> !qtx.wire_array<1>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%q0 = qtx.alloca : !qtx.wire
%i = arith.constant 0 : index
%q1, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>
// expected-error@+1 {{cannot yield 2 wires back to an array that only has 1 dead wire(s)}}
%e = qtx.array_yield %q0, %q1 to %new_array : !qtx.wire_array<1, dead = 1> -> !qtx.wire_array<1>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
%q0, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>
// expected-error@+1 {{must return an array with same size of the input array}}
%e = qtx.array_yield %q0 to %new_array : !qtx.wire_array<1, dead = 1> -> !qtx.wire_array<2>

// -----

%array = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
%q0, %new_array = qtx.array_borrow %i from %array : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>
// expected-error@+1 {{must return an array with 0 dead wire(s)}}
%e = qtx.array_yield %q0 to %new_array : !qtx.wire_array<1, dead = 1> -> !qtx.wire_array<1, dead = 1>

// -----

%a = qtx.alloca : !qtx.wire_array<1>
%i = arith.constant 0 : index
%w0, %a1 = qtx.array_borrow %i from %a : index from !qtx.wire_array<1> -> !qtx.wire, !qtx.wire_array<1, dead = 1>
// expected-error@+1 {{must not deallocate a wire array with dead wires}}
qtx.dealloc %a1 : !qtx.wire_array<1, dead = 1>

// -----

%a = qtx.alloca : !qtx.wire_array<2>
%i = arith.constant 0 : index
%w0, %a1 = qtx.array_borrow %i from %a : index from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire_array<2, dead = 1>
// expected-error@+1 {{must not split a wire array with dead wires}}
%w1, %w2 = qtx.array_split %a1 : !qtx.wire_array<2, dead = 1>

// -----

%a = qtx.alloca : !qtx.wire_array<2>
%i = arith.constant 0 : index
%w0, %a1 = qtx.array_borrow %i from %a : index from !qtx.wire_array<2> -> !qtx.wire, !qtx.wire_array<2, dead = 1>
// expected-error@+1 {{cannot measure an array with dead wires}}
%bit, %a2 = qtx.mz %a1 : !qtx.wire_array<2, dead = 1> -> <vector<2xi1>> !qtx.wire_array<2, dead = 1>

// -----

%a0 = qtx.alloca : !qtx.wire_array<3>
// expected-error@+1 {{must return a vector capable of holding all wires being measured}}
%bit, %a1 = qtx.mz %a0 : !qtx.wire_array<3> -> <vector<2xi1>> !qtx.wire_array<3>

// -----

%a0 = qtx.alloca : !qtx.wire_array<1>
// expected-error@+1 {{must return either a `i1` or `vector<1xi1>` , when measuring a wire or an array with one wire}}
%bit, %a1 = qtx.mz %a0 : !qtx.wire_array<1> -> <vector<2xi1>> !qtx.wire_array<1>
