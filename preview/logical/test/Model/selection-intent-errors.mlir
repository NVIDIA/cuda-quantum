// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  %accept = arith.constant true
  // expected-error @+1 {{mode must be require, condition_results, or abort_on}}
  qlx.selection %accept {mode = "retry_forever"} : i1
}

// -----

module {
  %abort = arith.constant true
  // expected-error @+1 {{accept_when disagrees with the selection mode}}
  fabric.selection %abort {mode = "abort_on", accept_when = true} : i1
}
