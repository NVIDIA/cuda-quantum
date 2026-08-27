# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for downstream MLIR dialect registration via entry points."""

# RUN: PYTHONPATH=../../ pytest -rP %s

from unittest.mock import MagicMock, patch

from cudaq.mlir._mlir_libs import _site_initialize_1
from cudaq.mlir.ir import DialectRegistry


def test_site_initialize_dispatches_mlir_dialect_entry_points():
    registry = DialectRegistry()
    seen = []

    fake_ep = MagicMock()
    fake_ep.load.return_value = lambda reg: seen.append(reg)

    with patch.object(_site_initialize_1,
                      "entry_points",
                      return_value=[fake_ep]) as mock_eps:
        _site_initialize_1.register_dialects(registry)

    mock_eps.assert_called_once_with(group="cudaq.mlir_dialects")
    fake_ep.load.assert_called_once_with()
    assert seen == [registry]
