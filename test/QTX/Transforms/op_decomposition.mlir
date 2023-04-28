// ========================================================================== //
// Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt --qtx-op-decomposition %s | FileCheck %s

module {

  // CHECK-LABEL: qtx.circuit @trivial_ccz() {
  // CHECK:         %[[VAL_0:.*]] = alloca : !qtx.wire
  // CHECK:         %[[VAL_1:.*]] = alloca : !qtx.wire
  // CHECK:         %[[VAL_2:.*]] = alloca : !qtx.wire
  // CHECK:         %[[VAL_3:.*]] = h %[[VAL_0]] : !qtx.wire
  // CHECK:         %[[VAL_4:.*]] = h %[[VAL_1]] : !qtx.wire
  // CHECK:         %[[VAL_5:.*]] = h %[[VAL_2]] : !qtx.wire
  // CHECK:         %[[VAL_6:.*]] = x {{\[}}%[[VAL_4]]] %[[VAL_5]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_7:.*]] = t<adj> %[[VAL_6]] : !qtx.wire
  // CHECK:         %[[VAL_8:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_7]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_9:.*]] = t %[[VAL_8]] : !qtx.wire
  // CHECK:         %[[VAL_10:.*]] = x {{\[}}%[[VAL_4]]] %[[VAL_9]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_11:.*]] = t<adj> %[[VAL_10]] : !qtx.wire
  // CHECK:         %[[VAL_12:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_11]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_13:.*]] = t %[[VAL_12]] : !qtx.wire
  // CHECK:         %[[VAL_14:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_4]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_15:.*]] = t<adj> %[[VAL_14]] : !qtx.wire
  // CHECK:         %[[VAL_16:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_15]] : [!qtx.wire] !qtx.wire
  // CHECK:         %[[VAL_17:.*]] = t %[[VAL_16]] : !qtx.wire
  // CHECK:         %[[VAL_18:.*]] = t %[[VAL_3]] : !qtx.wire
  // CHECK:         %[[VAL_19:.*]] = h %[[VAL_18]] : !qtx.wire
  // CHECK:         %[[VAL_20:.*]] = h %[[VAL_17]] : !qtx.wire
  // CHECK:         %[[VAL_21:.*]] = h %[[VAL_13]] : !qtx.wire
  // CHECK:         return
  // CHECK:       }
  qtx.circuit @trivial_ccz() {
    %a_0 = alloca : !qtx.wire
    %b_0 = alloca : !qtx.wire
    %t_0 = alloca : !qtx.wire

    %a_1 = h %a_0 : !qtx.wire
    %b_1 = h %b_0 : !qtx.wire
    %t_1 = h %t_0 : !qtx.wire
    %t_2 = z [%a_1, %b_1] %t_1 : [!qtx.wire, !qtx.wire] !qtx.wire
    %a_2 = h %a_1 : !qtx.wire
    %b_2 = h %b_1 : !qtx.wire
    %t_3 = h %t_2 : !qtx.wire
    return
  }

  // CHECK-LABEL:   qtx.circuit @trivial_ccx(
  // CHECK-SAME:                             %[[VAL_0:.*]]: !qtx.wire, %[[VAL_1:.*]]: !qtx.wire, %[[VAL_2:.*]]: !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire) {
  // CHECK:           %[[VAL_3:.*]] = h %[[VAL_0]] : !qtx.wire
  // CHECK:           %[[VAL_4:.*]] = h %[[VAL_1]] : !qtx.wire
  // CHECK:           %[[VAL_5:.*]] = h %[[VAL_2]] : !qtx.wire
  // CHECK:           %[[VAL_6:.*]] = h %[[VAL_5]] : !qtx.wire
  // CHECK:           %[[VAL_7:.*]] = x {{\[}}%[[VAL_4]]] %[[VAL_6]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_8:.*]] = t<adj> %[[VAL_7]] : !qtx.wire
  // CHECK:           %[[VAL_9:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_8]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_10:.*]] = t %[[VAL_9]] : !qtx.wire
  // CHECK:           %[[VAL_11:.*]] = x {{\[}}%[[VAL_4]]] %[[VAL_10]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_12:.*]] = t<adj> %[[VAL_11]] : !qtx.wire
  // CHECK:           %[[VAL_13:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_12]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_14:.*]] = t %[[VAL_13]] : !qtx.wire
  // CHECK:           %[[VAL_15:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_4]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_16:.*]] = t<adj> %[[VAL_15]] : !qtx.wire
  // CHECK:           %[[VAL_17:.*]] = x {{\[}}%[[VAL_3]]] %[[VAL_16]] : [!qtx.wire] !qtx.wire
  // CHECK:           %[[VAL_18:.*]] = t %[[VAL_17]] : !qtx.wire
  // CHECK:           %[[VAL_19:.*]] = t %[[VAL_3]] : !qtx.wire
  // CHECK:           %[[VAL_20:.*]] = h %[[VAL_14]] : !qtx.wire
  // CHECK:           %[[VAL_21:.*]] = h %[[VAL_19]] : !qtx.wire
  // CHECK:           %[[VAL_22:.*]] = h %[[VAL_18]] : !qtx.wire
  // CHECK:           %[[VAL_23:.*]] = h %[[VAL_20]] : !qtx.wire
  // CHECK:           return %[[VAL_21]], %[[VAL_22]], %[[VAL_23]] : !qtx.wire, !qtx.wire, !qtx.wire
  // CHECK:         }
  qtx.circuit @trivial_ccx(%a_0: !qtx.wire, %b_0: !qtx.wire, %t_0: !qtx.wire) -> (!qtx.wire, !qtx.wire, !qtx.wire) {
    %a_1 = h %a_0 : !qtx.wire
    %b_1 = h %b_0 : !qtx.wire
    %t_1 = h %t_0 : !qtx.wire
    %t_2 = x [%a_1, %b_1] %t_1 : [!qtx.wire, !qtx.wire] !qtx.wire
    %a_2 = h %a_1 : !qtx.wire
    %b_2 = h %b_1 : !qtx.wire
    %t_3 = h %t_2 : !qtx.wire
    return %a_2, %b_2, %t_3 : !qtx.wire, !qtx.wire, !qtx.wire
  }

  // CHECK-LABEL: qtx.circuit @oracle
  qtx.circuit @oracle() {
    %0 = alloca : !qtx.wire
    %1 = alloca : !qtx.wire
    %2 = alloca : !qtx.wire
    %3 = alloca : !qtx.wire
    %4 = alloca : !qtx.wire
    %5 = x [%1, %2] %4 : [!qtx.wire, !qtx.wire] !qtx.wire
    %6 = x [%2, %3] %5 : [!qtx.wire, !qtx.wire] !qtx.wire
    %7 = x [%0, %1] %3 : [!qtx.wire, !qtx.wire] !qtx.wire
    %8 = x [%2, %7] %6 : [!qtx.wire, !qtx.wire] !qtx.wire
    %9 = x [%0, %1] %7 : [!qtx.wire, !qtx.wire] !qtx.wire
    %10 = x [%0, %9] %8 : [!qtx.wire, !qtx.wire] !qtx.wire
    %11 = x [%9, %2] %10 : [!qtx.wire, !qtx.wire] !qtx.wire
    %12 = x [%0, %1] %2 : [!qtx.wire, !qtx.wire] !qtx.wire
    %13 = x [%9, %12] %11 : [!qtx.wire, !qtx.wire] !qtx.wire
    %14 = x [%0, %1] %12 : [!qtx.wire, !qtx.wire] !qtx.wire
    %15 = x [%9, %1] %13 : [!qtx.wire, !qtx.wire] !qtx.wire
    %16 = x [%0, %14] %1 : [!qtx.wire, !qtx.wire] !qtx.wire
    %17 = x [%9, %16] %15 : [!qtx.wire, !qtx.wire] !qtx.wire
    %18 = x [%0, %14] %16 : [!qtx.wire, !qtx.wire] !qtx.wire
    %19 = x [%9, %0] %17 : [!qtx.wire, !qtx.wire] !qtx.wire
    %20 = x [%18, %14] %0 : [!qtx.wire, !qtx.wire] !qtx.wire
    %21 = x [%9, %20] %19 : [!qtx.wire, !qtx.wire] !qtx.wire
    %22 = x [%18, %14] %20 : [!qtx.wire, !qtx.wire] !qtx.wire
    return
  }

}
