# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Availability filtering for cudaq.get_targets / has_target."""

from pathlib import Path

import cudaq


def test_unavailable_target_hidden_until_requested(tmp_path: Path):
    pkg = tmp_path / "needs-gpu-pkg"
    (pkg / "targets").mkdir(parents=True)
    (pkg / "lib").mkdir()
    (pkg / "targets" / "needs-gpu-avail.yml").write_text(
        "version: 1\n"
        "name: needs-gpu-avail\n"
        'description: "Requires a GPU for availability filtering."\n'
        "gpu-requirements: true\n"
        "config:\n"
        "  library-mode: true\n")

    cudaq.register_backend_path(str(pkg))

    if cudaq.num_available_gpus() == 0:
        assert not cudaq.has_target("needs-gpu-avail")
        assert cudaq.has_target("needs-gpu-avail", include_unavailable=True)
        names = [t.name for t in cudaq.get_targets()]
        assert "needs-gpu-avail" not in names
        hidden = [
            t for t in cudaq.get_targets(include_unavailable=True)
            if t.name == "needs-gpu-avail"
        ]
        assert len(hidden) == 1
        assert hidden[0].availability_diagnostic
    else:
        assert cudaq.has_target("needs-gpu-avail")
        names = [t.name for t in cudaq.get_targets()]
        assert "needs-gpu-avail" in names
