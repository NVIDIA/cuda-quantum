# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest

from cudaq.mlir.ir import Context, Module
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir._mlir_libs._quakeDialects import (cudaq_runtime,
                                                  register_all_dialects, quake,
                                                  cc)

from cudaq._compiler import optimization_kernels as kernels


def _context() -> Context:
    ctx = Context()
    register_all_dialects(ctx)
    quake.register_dialect(context=ctx)
    cc.register_dialect(context=ctx)
    return ctx


# The text comes from the real frontend, so it carries the realistic module
# wrapper a hand-written .qke snippet never would.
def test_lowered_text_has_real_frontend_shape():
    text = kernels.real_kernel_module_text("bell")
    assert "__nvqpp__mlirgen__bell" in text
    assert "quake.mangled_name_map" in text
    assert "hybridLaunchKernel" in text
    assert "cudaq-entrypoint" in text


# Loop-based kernels are only in-domain after the prepare pipeline unrolls their
# cc.loop.
def test_kernels_are_in_domain_after_prepare():
    ctx = _context()
    for name in kernels.real_kernel_names():
        module = Module.parse(kernels.real_kernel_module_text(name), ctx)
        assert module.operation.verify(), name
        PassManager.parse(kernels.DOMAIN_PREPARE_PIPELINE,
                          ctx).run(module.operation)
        pf = cudaq_runtime.preflight_bounded_unitary(module, 14)
        assert pf["supported"], (name, pf["rejections"])


# A Python `for` loop lowers to cc.loop, which is out of domain until unrolled,
# guarding the reason DOMAIN_PREPARE_PIPELINE is required, not optional.
def test_loop_kernel_is_out_of_domain_before_unroll():
    ctx = _context()
    module = Module.parse(kernels.real_kernel_module_text("ghz_linear"), ctx)
    pf = cudaq_runtime.preflight_bounded_unitary(module, 14)
    assert not pf["supported"]
    assert any(r["kind"] == "dynamic-control-flow" for r in pf["rejections"])


def test_module_text_rejects_unknown():
    with pytest.raises(ValueError):
        kernels.real_kernel_module_text("no_such_kernel")


def test_write_real_corpus_writes_one_file_per_kernel(tmp_path):
    paths = kernels.write_real_corpus(tmp_path)
    assert [p.name for p in paths
           ] == [f"{n}.qke" for n in kernels.real_kernel_names()]
    for path in paths:
        assert path.read_text().strip()
