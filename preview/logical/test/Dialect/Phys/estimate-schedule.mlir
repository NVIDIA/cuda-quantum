// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %S/../../Model/p3-physical-schedule.mlir --pass-pipeline='builtin.module(fabric-count{root=p2 device=device result=static},fabric-estimate-analytical{root=p2 counts=static device=device result=analytical p-phys=0.001 failure-budget=0.2},phys-estimate-schedule{schedule=events_schedule lower-tier=analytical result=scheduled})' | FileCheck %s
// RUN: not qlx-opt %S/../../Model/p3-physical-schedule.mlir --pass-pipeline='builtin.module(fabric-count{root=p2 device=device result=static},fabric-estimate-analytical{root=p2 counts=static device=device result=analytical p-phys=0.001 failure-budget=0.2},phys-estimate-schedule{schedule=missing lower-tier=analytical result=scheduled})' 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not qlx-opt %S/../../Model/p3-physical-schedule.mlir --phys-estimate-schedule='schedule=events_schedule result=scheduled' 2>&1 | FileCheck %s --check-prefix=NOLOWER

// CHECK: qlx.estimate_result @scheduled
// CHECK-SAME: data = {
// CHECK-DAG: active_resource_time_ns = 6.000000e+00
// CHECK-DAG: event_count = 5 : i64
// CHECK-DAG: makespan_ns = 3.000000e+00
// CHECK-DAG: peak_concurrency = 2 : i64
// CHECK-DAG: physical_resources = 2 : i64
// CHECK-DAG: utilization = 1.000000e+00
// CHECK-SAME: schema = "qlx.schedule-estimate/v2"
// CHECK-SAME: tier = "schedule"

// ERR: phys-estimate-schedule schedule @missing must resolve to phys.schedule
// NOLOWER: missing evidence: phys-estimate-schedule requires an exact
// NOLOWER-SAME: analytical lower-tier result
