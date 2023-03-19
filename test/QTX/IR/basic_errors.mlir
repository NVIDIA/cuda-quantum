// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt %s -split-input-file -verify-diagnostics

%w0 = qtx.alloca : !qtx.wire
%w1 = qtx.alloca : !qtx.wire
// expected-error@+1 {{failed to verify that the number of targets is equal to 1}}
%w2, %w3 = qtx.x %w0, %w1 : !qtx.wire, !qtx.wire

// -----

%w0 = qtx.alloca : !qtx.wire
// expected-error@+1 {{op failed to verify that the number of parameters is equal to 1}}
%w1 = qtx.rx %w0 : !qtx.wire

// -----

%w0 = qtx.alloca : !qtx.wire
%cst = arith.constant 3 : i32
// expected-error @+1 {{op operand #0 must be floating-point, but got 'i32'}}
%w1 = qtx.rx<%cst> %w0 : <i32> !qtx.wire

// -----

%w0 = qtx.alloca : !qtx.wire
// expected-error@+1 {{failed to verify that the number of targets is equal to 2}}
%w1 = qtx.swap %w0 : !qtx.wire

