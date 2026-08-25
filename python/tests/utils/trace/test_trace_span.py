# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

pytest.importorskip("cudaq")
from cudaq.util import trace


def test_span_enters_and_exits_cleanly():
    with trace.span("my_region"):
        value = 1 + 1
    assert value == 2


def test_span_nested_does_not_raise():
    with trace.span("outer"):
        with trace.span("inner"):
            pass


def test_span_propagates_exception():
    with pytest.raises(ValueError, match="inner failure"):
        with trace.span("region"):
            raise ValueError("inner failure")


def test_span_accepts_arbitrary_kwargs():
    with trace.span("region", key=1, other="x"):
        pass


def test_traced_decorator_emits_span_and_preserves_return():
    """@traced wraps the call in a span, preserves metadata, and returns
    the wrapped function's value. Exercises the captured-backend path so a
    future regression in wrapper plumbing shows up immediately."""
    backend = trace.ChromeBackend()
    trace.set_backend(backend)

    @trace.traced("traced.region")
    def add(a, b):
        """Add two numbers."""
        return a + b

    try:
        assert add(2, 3) == 5
    finally:
        trace.reset_backend()

    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."
    events = backend.to_dict()["traceEvents"]
    assert any(
        e.get("name") == "traced.region" and e.get("cat") == "python"
        for e in events)
