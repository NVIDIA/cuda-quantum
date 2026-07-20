# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# An input outside the bounded-unitary domain (it contains a measurement) must
# fail closed: `unsupported-domain` status, a structured reason naming the
# offending op, and a nonzero exit code. `not` asserts the nonzero exit while
# still piping stdout to FileCheck.

# RUN: PYTHONPATH=%cudaq_python_root not python3 -m cudaq._compiler.optimization_cli \
# RUN:   --input %S/Inputs/measurement.qke \
# RUN:   --candidate 'builtin.module(func.func(phase-folding))' | FileCheck %s

# CHECK-DAG: "status": "unsupported-domain"
# CHECK-DAG: measurement in kern
# CHECK-DAG: "equal_up_to_global_phase": false
