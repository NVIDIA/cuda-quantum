// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file --verify-diagnostics %s

module attributes {qlx.profiles = ["p2n"]} {
  // expected-error @+1 {{metadata with input_p1 requires qec_selection_sha256}}
  fabric.protocol @missing_commitment : () -> () attributes {
    metadata = {input_p1 = "placed_kernel"}
  } {
    fabric.protocol_return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  // expected-error @+1 {{metadata input_p1 must be a nonempty string}}
  fabric.protocol @malformed_input : () -> () attributes {
    metadata = {
      input_p1 = 7 : i64,
      qec_selection_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  } {
    fabric.protocol_return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  // expected-error @+1 {{metadata qec_selection_sha256 must be a string}}
  fabric.protocol @wrong_commitment_type : () -> () attributes {
    metadata = {
      input_p1 = "placed_kernel",
      qec_selection_sha256 = 7 : i64
    }
  } {
    fabric.protocol_return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  // expected-error @+1 {{metadata qec_selection_sha256 requires input_p1}}
  fabric.protocol @missing_input : () -> () attributes {
    metadata = {
      qec_selection_sha256 = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  } {
    fabric.protocol_return
  }
}

// -----

module attributes {qlx.profiles = ["p2n"]} {
  // expected-error @+1 {{metadata qec_selection_sha256 must be 'sha256:' followed by 64 lowercase hexadecimal digits}}
  fabric.protocol @malformed_commitment : () -> () attributes {
    metadata = {
      input_p1 = "placed_kernel",
      qec_selection_sha256 = "sha256:NOT-A-DIGEST"
    }
  } {
    fabric.protocol_return
  }
}
