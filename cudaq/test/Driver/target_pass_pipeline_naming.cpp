/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Regression test for a naming collision in `registerAllTargetPassPipelines()`
// (cudaq-opt.cpp): the registered pass pipeline name for a configuration-matrix
// entry is built by joining the target's name and the entry's name. Target and
// entry names routinely contain '-' themselves (e.g. "nvidia-mqpu-fp64",
// "single-gpu-fp32"), so joining with '-' let two distinct (target, entry)
// pairs collide on the same registered name -- e.g. target "xyz-bar" (no matrix
// entry) and target "xyz" with matrix entry "bar" both produced
// "target-pass-pipeline-xyz-bar". The fix joins with '.' instead, and also
// rejects any remaining collision outright rather than relying solely on the
// separator choice.

// RUN: split-file %s %t

// A target with its own hyphenated name and a *different* target whose
// configuration-matrix entry name, when joined with '-', would reproduce that
// same hyphenated name. Before the fix, this pair collided; now both pipelines
// register and are independently selectable.

// clang-format off
// RUN: cudaq-opt --register-target-pipelines=%t/xyz-bar.yml --register-target-pipelines=%t/xyz.yml --pass-pipeline='builtin.module(target-pass-pipeline-xyz-bar)' %t/empty.mlir -o /dev/null
// RUN: cudaq-opt --register-target-pipelines=%t/xyz-bar.yml --register-target-pipelines=%t/xyz.yml --pass-pipeline='builtin.module(target-pass-pipeline-xyz.bar)' %t/empty.mlir -o /dev/null
// clang-format on

// The collision detector is a safety net, not just the separator choice: a
// target literally named "xyz.bar" still collides with target "xyz"'s matrix
// entry "bar" (both produce "target-pass-pipeline-xyz.bar") and must be
// rejected with a clear diagnostic rather than silently double-registered.

// clang-format off
// RUN: not cudaq-opt --register-target-pipelines=%t/xyz.bar.yml --register-target-pipelines=%t/xyz.yml 2>&1 | FileCheck %s

//--- xyz-bar.yml
version: 1
name: xyz-bar
description: "Test target: hyphenated name."
config:
  target-pass-pipeline: "canonicalize"

//--- xyz.yml
version: 1
name: xyz
description: "Test target: configuration-matrix entry named 'bar'."
target-arguments:
  - key: option
    required: false
    type: option-flags
    help-string: "Specify the target options."
configuration-matrix:
  - name: bar
    option-flags: [qpp]
    default: true
    config:
      target-pass-pipeline: "canonicalize"

//--- xyz.bar.yml
version: 1
name: xyz.bar
description: "Test target: name literally reproduces the '.'-joined form."
config:
  target-pass-pipeline: "canonicalize"

//--- empty.mlir
module {}

// CHECK: duplicate target pass pipeline name 'target-pass-pipeline-xyz.bar'
