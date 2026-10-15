/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef TEST_FTQC_OPS_TD
#define TEST_FTQC_OPS_TD

include "test_ftqc/TestFTQCDialect.td"
include "CUDAQLogical/Interfaces/PhysicalInterfaces.td"

class TestFTQC_Op<string mnemonic, list<Trait> traits = []>
    : Op<TestFTQC_Dialect, mnemonic, traits>;

def TestFTQC_MeasureRoundOp : TestFTQC_Op<"measure_round", [
    DeclareOpInterfaceMethods<CUDAQLogical_PhysicalMeasurementOpInterface>
  ]> {
  let summary = "target-specific P3 measurement extension";
  let arguments = (ins AnyType:$input, StrAttr:$record_id,
      OptionalAttr<StrAttr>:$event_id, OptionalAttr<StrAttr>:$test_mode);
  let results = (outs AnyType:$output, AnyType:$record);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}

def TestFTQC_MeasureWithPayloadOp
    : TestFTQC_Op<"measure_with_payload",
                  [DeclareOpInterfaceMethods<
                      CUDAQLogical_PhysicalMeasurementOpInterface>]> {
  let summary = "malformed measurement carrying an unmodelled linear value";
  let arguments = (ins AnyType:$input, AnyType:$payload, StrAttr:$record_id);
  let results = (outs AnyType:$output, AnyType:$record);
  let hasVerifier = 1;
}

def TestFTQC_MeasureWithExtraRecordOp
    : TestFTQC_Op<"measure_with_extra_record",
                  [DeclareOpInterfaceMethods<
                      CUDAQLogical_PhysicalMeasurementOpInterface>]> {
  let summary = "malformed measurement with an unreported record";
  let arguments = (ins AnyType:$input, StrAttr:$record_id);
  let results = (outs AnyType:$output, AnyType:$record, AnyType:$extra_record);
  let hasVerifier = 1;
}

def TestFTQC_OpaqueMeasureOp : TestFTQC_Op<"opaque_measure", []> {
  let summary = "negative test operation with no physical event contract";
  let arguments = (ins AnyType:$input, StrAttr:$record_id);
  let results = (outs AnyType:$output, AnyType:$record);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
}

#endif // TEST_FTQC_OPS_TD
