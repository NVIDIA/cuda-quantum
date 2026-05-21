# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq

_DETECTOR_EMPTY = ("detector requires at least one "
                   "cudaq.measure_handle argument")
_OBSERVABLE_EMPTY = ("logical_observable requires at least one "
                     "cudaq.measure_handle argument")
_OBS_IDX_NOT_INT = ("logical_observable requires observable_index "
                    "to be an integer literal")
_OBS_IDX_RANGE = ("logical_observable observable_index must be in "
                  "the range [0, 2^63 - 1]")


@pytest.fixture(autouse=True)
def reset_run_clear():
    cudaq.reset_target()
    yield
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


# ---------------------------------------------------------------------------
# Host-scope invocation fails loudly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op,args", [
    (cudaq.detector, ()),
    (cudaq.logical_observable, ()),
    (cudaq.detectors, ([], [])),
])
def test_host_raises(op, args):
    with pytest.raises(RuntimeError) as e:
        op(*args)
    assert "can be used only in CUDA-Q kernels" in str(e.value)


# ---------------------------------------------------------------------------
# Bridge-level rejections: empty operand lists and wrong arity for the
# binary op.
# ---------------------------------------------------------------------------


def test_detector_no_args_rejected():

    @cudaq.kernel
    def k():
        cudaq.detector()

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _DETECTOR_EMPTY in str(e.value)


def test_observable_no_args_rejected():

    @cudaq.kernel
    def k():
        cudaq.logical_observable()

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _OBSERVABLE_EMPTY in str(e.value)


def test_detectors_one_arg_rejected():

    @cudaq.kernel
    def k():
        qs = cudaq.qvector(3)
        handles = mz(qs)
        cudaq.detectors(handles)

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert ("cudaq.detectors takes exactly two "
            "list[cudaq.measure_handle] arguments") in str(e.value)


# ---------------------------------------------------------------------------
# `observable_index` must be a Python integer literal in `[0, 2^63 - 1]`.
# ---------------------------------------------------------------------------


def test_observable_runtime_idx_rejected():

    @cudaq.kernel
    def k(idx: int):
        qs = cudaq.qvector(2)
        handles = mz(qs)
        cudaq.logical_observable(handles, observable_index=idx)

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _OBS_IDX_NOT_INT in str(e.value)


_MODULE_OBS_IDX = 5


def test_observable_module_var_rejected():

    @cudaq.kernel
    def k():
        qs = cudaq.qvector(3)
        handles = mz(qs)
        cudaq.logical_observable(handles, observable_index=_MODULE_OBS_IDX)

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _OBS_IDX_NOT_INT in str(e.value)


def test_observable_negative_idx_rejected():

    @cudaq.kernel
    def k():
        qs = cudaq.qvector(2)
        handles = mz(qs)
        cudaq.logical_observable(handles, observable_index=-1)

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _OBS_IDX_RANGE in str(e.value)


def test_observable_overflow_idx_rejected():

    @cudaq.kernel
    def k():
        qs = cudaq.qvector(2)
        handles = mz(qs)
        cudaq.logical_observable(handles, observable_index=9223372036854775808)

    with pytest.raises(RuntimeError) as e:
        k.compile()
    assert _OBS_IDX_RANGE in str(e.value)


# ===========================================================================
# `cudaq.make_kernel()` programmatic builder surface.
# ===========================================================================


def test_builder_detector_no_args_rejected():
    kernel = cudaq.make_kernel()
    with pytest.raises(RuntimeError) as e:
        kernel.detector()
    assert _DETECTOR_EMPTY in str(e.value)


def test_builder_observable_no_args_rejected():
    kernel = cudaq.make_kernel()
    with pytest.raises(RuntimeError) as e:
        kernel.logical_observable()
    assert _OBSERVABLE_EMPTY in str(e.value)


def test_builder_detector_non_handle_rejected():
    kernel = cudaq.make_kernel()
    with pytest.raises(RuntimeError) as e:
        kernel.detector(42)
    assert ("kernel.detector arguments must be QuakeValue "
            "measurement handles") in str(e.value)


def test_builder_detectors_scalar_arg_rejected():
    kernel = cudaq.make_kernel()
    q0 = kernel.qalloc()
    q1 = kernel.qalloc()
    h0 = kernel.mz(q0)
    h1 = kernel.mz(q1)
    with pytest.raises(RuntimeError) as e:
        kernel.detectors(h0, h1)
    assert ("kernel.detectors arguments must each be a "
            "list[cudaq.measure_handle]") in str(e.value)


@pytest.fixture
def builder_kernel_and_handles():
    kernel = cudaq.make_kernel()
    qs = kernel.qalloc(2)
    hs = kernel.mz(qs)
    return kernel, hs


def test_builder_observable_non_int_idx_rejected(builder_kernel_and_handles):
    kernel, hs = builder_kernel_and_handles
    with pytest.raises(RuntimeError) as e:
        kernel.logical_observable(hs, observable_index="not-an-int")
    assert _OBS_IDX_NOT_INT in str(e.value)


@pytest.mark.parametrize("idx", [-1, 9223372036854775808])
def test_builder_observable_idx_out_of_range(builder_kernel_and_handles, idx):
    kernel, hs = builder_kernel_and_handles
    with pytest.raises(RuntimeError) as e:
        kernel.logical_observable(hs, observable_index=idx)
    assert _OBS_IDX_RANGE in str(e.value)


# leave for gdb debugging
if __name__ == "__main__":
    import os
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
