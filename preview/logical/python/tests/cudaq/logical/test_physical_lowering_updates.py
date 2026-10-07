# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Focused detector-free P3 projection regressions."""

from types import SimpleNamespace

import pytest

import cudaq.mlir.ir as mlir_ir
from cudaq.logical.compiler.physical_lower import _Carrier, _P2ToP3, _Patch


def _operation(name, **attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def test_release_history_retains_every_predecessor_for_reused_index():
    lowerer = _P2ToP3.__new__(_P2ToP3)
    lowerer._exclusive_path = []
    lowerer._released_index_events = {"qubits": {}}
    allocation = SimpleNamespace(resource_class="qubits", indices=(3,))

    lowerer._record_release(allocation, "release0")
    lowerer._record_release(allocation, "release1")

    assert lowerer._resource_predecessors("qubits", (3,)) == (
        "release0",
        "release1",
    )


def test_native_instrument_reuses_the_device_declaration():
    lowerer = _P2ToP3.__new__(_P2ToP3)
    declaration = _operation("phys.instrument", sym_name="measure_z_instrument")

    class _Transaction:

        @staticmethod
        def find_symbol(name, kind):
            assert (name, kind) == ("measure_z_instrument", "phys.instrument")
            return declaration

        @staticmethod
        def materialize(_instrument):
            raise AssertionError("device instrument was rematerialized")

    lowerer.transaction = _Transaction()
    instrument = SimpleNamespace(name="measure_z_instrument")

    assert lowerer._native_instrument_symbol(instrument) == (
        "measure_z_instrument")


def test_patch_transform_borrows_and_releases_structural_scratch():
    lowerer = _P2ToP3.__new__(_P2ToP3)
    transform = _operation(
        "fabric.patch_transform",
        source="E0",
        destination="E1",
        frame_partitions=(SimpleNamespace(name="data", attr=2),),
        source_support=(0,),
        destination_support=(0,),
    )
    lowerer.symbols = {
        "T": transform,
        "E1": _operation("fabric.encoding", code="C1"),
    }
    lowerer.codes = {"C1": (("data", 1),)}
    lowerer.device = SimpleNamespace(physical=SimpleNamespace(
        resource_classes=(SimpleNamespace(name="qubits", count=2),)))
    lowerer._transform_frames = {}
    lowerer._allocated_indices = {"qubits": {0}}
    lowerer._ever_allocated_indices = {"qubits": {0}}
    lowerer._resource_predecessors = lambda _name, _indices: ()
    lowerer._scratch = {}
    released = []

    def scratch(resource_class, index, template):
        carrier = _Carrier(
            f"q{index}",
            f"q{index}",
            resource_class,
            template.region,
            template.qec_region,
            template.physical_binding,
        )
        lowerer._scratch[(resource_class, index,
                          template.physical_binding)] = carrier
        return carrier

    lowerer._scratch_carrier = scratch
    lowerer._release_route_scratch = lambda values: released.extend(
        carrier.resource for carrier in values.values())
    lowerer._allocation_by_patch = {
        "patch0": SimpleNamespace(resources=("q0",))
    }
    patch = _Patch(
        "C0",
        "E0",
        {"data": [_Carrier("q0", "q0", "qubits", "memory", None, "binding")]},
        "memory",
        "patch0",
        physical_binding="binding",
    )
    begin = _operation("fabric.transform_begin", transform="T")
    end = _operation("fabric.transform_end", transform="T")

    frame = lowerer._transform_begin(begin, patch)
    assert tuple(carrier.resource for carrier in frame.all()) == ("q0", "q1")
    result = lowerer._transform_end(end, frame)

    assert result.code == "C1"
    assert result.encoding == "E1"
    assert tuple(carrier.resource for carrier in result.all()) == ("q0",)
    assert released == ["q1"]
    assert not lowerer._transform_frames
    assert not lowerer._scratch


def test_encoded_plus_fails_closed_without_logical_x_representatives():
    lowerer = _P2ToP3.__new__(_P2ToP3)
    lowerer.symbols = {
        "Repetition":
            _operation(
                "fabric.code",
                hx=(),
                hz=((0, 1), (1, 2)),
                lz=((0,),),
            )
    }
    patch = _Patch(
        "Repetition",
        "encoding",
        {
            "data": [
                _Carrier(f"q{index}", f"q{index}", "qubits")
                for index in range(3)
            ]
        },
        "memory",
        "patch0",
    )

    with pytest.raises(
            ValueError,
            match="requires explicit logical-X representatives",
    ):
        lowerer._prepare(patch, "plus", encode=True)


def test_css_zero_encoder_spans_the_final_stabilizer_row_space():
    lowerer = _P2ToP3.__new__(_P2ToP3)
    lowerer.context = mlir_ir.Context()
    lowerer.symbols = {
        "CounterexampleCSS":
            _operation(
                "fabric.code",
                hx=((0, 1), (0, 1, 2, 3)),
            )
    }
    lowerer._event_id = lambda prefix: f"{prefix}0"
    lowerer._emit = lambda name, *, operands, results, attributes: (
        SimpleNamespace(results=tuple(
            SimpleNamespace(type=result_type) for result_type in results)))
    h_controls = []
    cx_operations = []

    def apply(carriers, action):
        assert action == "h"
        h_controls.append(int(carriers[0].resource.removeprefix("q")))
        return carriers

    def route(control, target, action):
        assert action == "cx"
        control_index = int(control.resource.removeprefix("q"))
        target_index = int(target.resource.removeprefix("q"))
        cx_operations.append((control_index, target_index))
        return control, target

    lowerer._apply_carriers = apply
    lowerer._routed_pair = route
    patch = _Patch(
        "CounterexampleCSS",
        "encoding",
        {
            "data": [
                _Carrier(
                    SimpleNamespace(type=f"q{index}"),
                    f"q{index}",
                    "qubits",
                ) for index in range(5)
            ]
        },
        "memory",
        "patch0",
    )

    lowerer._prepare(patch, "zero", encode=True)

    generated = [1 << control for control in h_controls]
    for control, target in cx_operations:
        generated = [
            row ^ (1 << target) if (row >> control) & 1 else row
            for row in generated
        ]
    intended = {0b00011, 0b01111}

    def span(rows):
        values = {0}
        for row in rows:
            values |= {value ^ row for value in tuple(values)}
        return values

    assert span(generated) == span(intended)
