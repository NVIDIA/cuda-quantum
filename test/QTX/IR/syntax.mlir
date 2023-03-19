// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// This file is only here to document QTX syntax style and rationale
// RUN: cudaq-opt %s | cudaq-opt | FileCheck %s

module {

  // CHECK-LABEL:   qtx.circuit @syntax01<
  // CHECK-SAME:                          %[[VAL_0:.*]]: f64>(
  // CHECK-SAME:                          %[[VAL_1:.*]]: !qtx.wire,
  // CHECK-SAME:                          %[[VAL_2:.*]]: !qtx.wire) -> <i1>(!qtx.wire, !qtx.wire) {
  // CHECK:           %[[VAL_7:.*]] = t %[[VAL_1]] : !qtx.wire
  // CHECK:           %[[VAL_8:.*]]:2 = swap %[[VAL_7]], %[[VAL_2]] : !qtx.wire, !qtx.wire
  // CHECK:           %[[VAL_9:.*]] = t<adj> %[[VAL_8]]#0 : !qtx.wire
  // CHECK:           %[[VAL_10:.*]] = rz<%[[VAL_0]]> %[[VAL_8]]#1 : <f64> !qtx.wire
  // CHECK:           %[[VAL_11:.*]] = rz<adj, %[[VAL_0]]> %[[VAL_10]] : <f64> !qtx.wire
  // CHECK:           %[[VAL_12:.*]] = x {{\[}}%[[VAL_9]]] %[[VAL_11]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = mz %[[VAL_12]] : !qtx.wire -> <i1> !qtx.wire
  // CHECK:           return <%[[VAL_13]]> %[[VAL_9]], %[[VAL_14]] : <i1> !qtx.wire, !qtx.wire
  // CHECK:         }

  // We decided to separate classical parameters and quantum parameters in the
  // signature of QTX circuits, which makes retriving targets much easier.  The
  // syntax
  //                     +-- classical arguments
  //                     |
  //                     |                   +-- classical result types
  //                  v-----v           v--------v
  // qtx.circuit @name<cargs>(qargs) -> <cresults>(qresults)
  //                         ^-----^              ^--------^
  //                            |                      + quantum result types
  //                            |
  //                            +-- quantum arguments
  //
  // Note: the `qresults` types must match `qargs` types, so the former does
  // not provides additional information.
  //
  // Example:
  qtx.circuit @syntax01<%angle: f64>(%w0: !qtx.wire, %w1: !qtx.wire) -> <i1>(!qtx.wire, !qtx.wire) {

    // A simple operation has the following syntax:
    //
    //        + list of results
    //        |                       + list of targets
    //   v---------v             v---------v
    //  %r0, ..., %rn = op-name %t0, ..., %tn : type(%t0), ..., type(%tn)
    //                                             ^---------------^
    //                                list of targes' type +
    //
    // NOTE: We don't need to explicitly specify the results' type because we
    //       can infer them from the targets.
    //
    // Example:
    %0 = t %w0 : !qtx.wire
    %1:2 = swap %0, %w1 : !qtx.wire, !qtx.wire

    // We can parametrize an operation with an attribute that make it the adjoint
    //    vvvvv
    %2 = t<adj> %1#0 : !qtx.wire

    // Some operations require classical parameter(s)
    //      vvvvvv
    %3 = rz<%angle> %1#1 : <f64> !qtx.wire
    //                     ^^^^^
    // Note that the parameters' type also go right to the colon

    // When we mix classical parameters and the adjoint attribute, the latter
    // must come first
    //      vvv
    %4 = rz<adj, %angle> %3 : <f64> !qtx.wire

    // We can add controls
    //     vvvv      vvvvvvvvvvv
    %5 = x [%2] %4 : [!qtx.wire] !qtx.wire

    // Measurement along z-axis
    %bit, %6 = mz %5 : !qtx.wire -> <i1> !qtx.wire

    return <%bit> %2, %6 : <i1> !qtx.wire, !qtx.wire
  }
}

