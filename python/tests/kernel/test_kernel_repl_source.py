# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Regression tests for GitHub issue #2593: decorating a function with
`@cudaq.kernel` in the standard Python REPL previously raised an opaque
`OSError: could not get source code`. The fix replaces that with a
`RuntimeError` whose message explains the cause and suggests workarounds.
"""

import linecache

import pytest

import cudaq
from cudaq.kernel.analysis import FetchDepFuncsSourceCode
from cudaq.kernel.utils import get_function_source_or_raise


@pytest.fixture(autouse=True)
def _clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def _make_synthetic_function(src, name, filename):
    """
    Compile `src` with `filename` as its source-of-record (mimicking what
    CPython does when it executes code typed at the REPL). Returns the
    named callable from the resulting namespace.

    The filename must not already be cached in `linecache` — otherwise
    `inspect.getsource` could succeed unexpectedly and produce a false
    negative for the tests below.
    """
    assert filename not in linecache.cache, (
        f"linecache already has an entry for {filename!r}; pick a unique name")
    code = compile(src, filename, 'exec')
    ns = {}
    exec(code, ns)
    return ns[name]


def test_repl_decoration_raises_clear_error():
    """
    Direct reproduction of issue #2593: a function with `<stdin>` as its
    source filename cannot be compiled, but the error must name the
    function and point at Jupyter/file workarounds instead of surfacing a
    raw `OSError`.
    """
    fn = _make_synthetic_function(
        "def my_repl_kernel(n: int):\n    pass\n",
        name='my_repl_kernel',
        filename='<stdin>',
    )

    with pytest.raises(RuntimeError) as excinfo:
        cudaq.kernel(fn)

    msg = str(excinfo.value)
    assert 'my_repl_kernel' in msg
    assert 'REPL' in msg
    assert 'Jupyter' in msg
    # Original OSError preserved for debugging.
    assert isinstance(excinfo.value.__cause__, OSError)


def test_synthetic_filename_raises_non_repl_message():
    """
    A function whose source filename is synthetic but not the REPL
    sentinel (e.g., `<generated>`) produces the non-file-context message,
    not the REPL-specific one.
    """
    fn = _make_synthetic_function(
        "def generated_kernel(n: int):\n    pass\n",
        name='generated_kernel',
        filename='<generated-test-src>',
    )

    with pytest.raises(RuntimeError) as excinfo:
        get_function_source_or_raise(fn)

    msg = str(excinfo.value)
    assert 'generated_kernel' in msg
    assert '<generated-test-src>' in msg
    # Must not misidentify this as a REPL case.
    assert 'REPL' not in msg


def test_dep_fetch_raises_clear_error_for_repl_helper():
    """
    When a kernel calls a helper defined in the REPL, the dependency
    fetcher in `analysis.py` must surface the same clear diagnostic,
    naming the offending helper rather than blowing up with `OSError`.
    """
    repl_helper = _make_synthetic_function(
        "def repl_helper(x: int) -> int:\n    return x + 1\n",
        name='repl_helper',
        filename='<python-input-1>',
    )

    def parent_kernel(x: int) -> int:
        return repl_helper(x)

    # Inject the helper into the calling frame's locals so
    # FetchDepFuncsSourceCode can resolve it by name, then trigger the
    # dep fetch. The failure happens inside analysis.py, not the decorator.
    with pytest.raises(RuntimeError) as excinfo:
        FetchDepFuncsSourceCode.fetch(parent_kernel)

    msg = str(excinfo.value)
    assert 'repl_helper' in msg
    assert 'REPL' in msg
    assert isinstance(excinfo.value.__cause__, OSError)


def test_normal_function_still_compiles():
    """
    Regression guard: ensure the error-path wrapping did not break the
    ordinary success path. A kernel defined in this test file (which
    `inspect.getsource` can read) must compile without raising.
    """

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    result = cudaq.sample(bell_pair, shots_count=100)
    # The test passes if decoration and sampling succeed; specific counts
    # are irrelevant here.
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
