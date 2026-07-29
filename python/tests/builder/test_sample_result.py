# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Unit tests for SampleResult construction and type interface.

These tests exercise the Python-constructible path (no kernel required):
counts constructor, annotations, Mapping protocol, counts property, and repr.
"""

import pytest

import cudaq


class TestCountsConstructor:

    def test_basic_construction(self):
        r = cudaq.SampleResult({"00": 512, "11": 488})
        assert r["00"] == 512
        assert r["11"] == 488

    def test_get_total_shots_sums_counts_when_no_annotation(self):
        r = cudaq.SampleResult({"00": 512, "11": 488})
        assert r.get_total_shots() == 1000

    def test_annotations_shots_does_not_affect_total_shots(self):
        r = cudaq.SampleResult({
            "00": 480,
            "11": 470
        },
                               annotations={"shots": 1000})
        assert r.get_total_shots() == 950  # always sum of counts
        assert r.annotations["shots"] == 1000  # read directly for original value

    def test_annotations_default_is_empty(self):
        r = cudaq.SampleResult({"0": 100})
        assert r.annotations == {}

    def test_annotations_roundtrip(self):
        r = cudaq.SampleResult({"0": 100},
                               annotations={
                                   "shots": 200,
                                   "tag": "test"
                               })
        assert r.annotations["shots"] == 200
        assert r.annotations["tag"] == "test"

    def test_empty_counts(self):
        r = cudaq.SampleResult({})
        assert len(r) == 0
        assert r.get_total_shots() == 0

    def test_single_bitstring(self):
        r = cudaq.SampleResult({"111": 1000})
        assert r["111"] == 1000
        assert r.get_total_shots() == 1000


class TestCountsProperty:

    def test_counts_returns_dict(self):
        r = cudaq.SampleResult({"00": 512, "11": 488})
        c = r.counts
        assert isinstance(c, dict)
        assert c["00"] == 512
        assert c["11"] == 488

    def test_counts_is_snapshot(self):
        r = cudaq.SampleResult({"00": 512})
        c = r.counts
        c["00"] = 0
        assert r["00"] == 512

    def test_counts_empty(self):
        r = cudaq.SampleResult({})
        assert r.counts == {}


class TestAnnotations:

    def test_annotations_attribute_exists(self):
        r = cudaq.SampleResult({"0": 1})
        assert hasattr(r, "annotations")

    def test_annotations_is_readonly(self):
        r = cudaq.SampleResult({"0": 1})
        with pytest.raises(AttributeError):
            r.annotations = {"custom": "value"}

    def test_shots_annotation_any_value_accepted(self):
        # annotations are stored as-is; any JSON-serialisable value is fine.
        cudaq.SampleResult({"0": 5}, annotations={"shots": 100})
        cudaq.SampleResult({"0": 5}, annotations={"shots": "hundred"})
        cudaq.SampleResult({"0": 5}, annotations={"shots": 100.5})


class TestMappingProtocol:

    def setup_method(self):
        self.r = cudaq.SampleResult({"00": 512, "11": 488})

    def test_contains_true(self):
        assert "00" in self.r

    def test_contains_false(self):
        assert "01" not in self.r

    def test_iter_yields_bitstrings(self):
        assert set(self.r) == {"00", "11"}

    def test_len(self):
        assert len(self.r) == 2

    def test_getitem(self):
        assert self.r["00"] == 512

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.r["01"]

    def test_items(self):
        assert dict(self.r.items()) == {"00": 512, "11": 488}

    def test_values(self):
        assert set(self.r.values()) == {512, 488}
