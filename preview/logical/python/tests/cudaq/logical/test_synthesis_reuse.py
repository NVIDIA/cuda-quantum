# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Repeated rotations retain the words obtained by synthesizing in isolation."""

import pytest

from cudaq.mlir import ir
from cudaq.mlir._mlir_libs import _qlxRuntime as runtime


def _rotation(name, angle, sign=1, precision=None, basis="Z", exact=False):
    parameters = [
        f"sign = {sign} : i64",
        f"x_mask = {int(basis in ('X', 'Y'))} : i64",
        f"z_mask = {int(basis in ('Y', 'Z'))} : i64",
    ]
    if precision is not None:
        parameters.append(f"precision = {precision:.17e} : f64")
    if exact:
        parameters.extend(
            ("angle_pi_numer = 1 : i64", "angle_pi_denom = 4 : i64"))
    return f"""
      qlx.program @{name} : () -> i1 attributes {{qlx.stage = "p0"}} {{
        %q = qlx.prepare "zero" {{allocation = 0 : i64, value_index = 0 : i64}}
          : !qlx.logical_qubit
        %angle = arith.constant {angle} : f64
        %r = qlx.apply #qlx.action<pauli_rotation>(%q, %angle)
          {{parameters = {{{', '.join(parameters)}}}}}
          : (!qlx.logical_qubit, f64) -> !qlx.logical_qubit
        %m = qlx.measure #qlx.pauli<Z> %r : !qlx.logical_qubit -> i1
        qlx.return %m : i1
      }}
    """


def _synthesize(programs, precision):
    module = ir.Module.parse("module {" + "\n".join(programs) + "}",
                             ir.Context())
    runtime.run_pass(
        module, f"qlx-synthesize-rotations{{precision={precision}}},"
        "qlx-verify-clifford-t")
    return [str(operation) for operation in module.body.operations]


@pytest.mark.parametrize("default_precision", [1e-3, 1e-8])
def test_repeated_rotations_match_independent_synthesis(default_precision):
    # Identical requests hit the cache; signs, angles and authored precision
    # must stay distinct. Basis changes must be applied at every use site.
    requests = [
        dict(angle=0.1234),
        dict(angle=0.1234),
        dict(angle=0.1234, basis="X"),
        dict(angle=0.1234, basis="Y"),
        dict(angle=0.1234, sign=-1),
        dict(angle=0.1234, sign=-1),
        dict(angle=0.1234, precision=1e-2),
        dict(angle=0.1234, precision=1e-6),
        dict(angle=0.1234, precision=1e-6),
        dict(angle=0.2468),
        # At this precision the rounded float needs approximation, but exact
        # rational-pi metadata must still select the exact T word.
        dict(angle=0.7853981633974483, precision=1e-18),
        dict(angle=0.7853981633974483, precision=1e-18, exact=True),
    ]
    programs = [
        _rotation(f"rotation_{index}", **request)
        for index, request in enumerate(requests)
    ]
    independent = [
        _synthesize([program], default_precision)[0] for program in programs
    ]
    assert _synthesize(programs, default_precision) == independent
