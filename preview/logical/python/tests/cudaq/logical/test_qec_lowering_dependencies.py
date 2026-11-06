# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Generated QEC manifests retain their complete dependency closure."""

import cudaq.logical as cql
import cudaq.mlir.ir as mlir_ir
import pytest

CODE = cql.codes.Steane


@cql.gadget(implements=cql.logical.h)
def dependency_h(block: cql.patch[CODE]) -> cql.patch[CODE]:
    return cql.h(block.data)


@cql.gadget(implements=cql.logical.x)
def dependency_x(block: cql.patch[CODE]) -> cql.patch[CODE]:
    return cql.x(block.data)


@pytest.mark.parametrize("reuse_dependency", (False, True))
def test_generated_dependencies_preserve_order_and_deduplicate(
        reuse_dependency):
    generated = {"lower_h": [], "lower_x": []}

    @cql.compiler.qec_lowering(objective=cql.logical.h,
                               codes=(CODE,),
                               dependencies=(dependency_h,))
    def lower_h(site, context):
        if reuse_dependency:
            return dependency_h

        @cql.protocol(implements=cql.logical.h)
        def generated_h(block: cql.patch[CODE]) -> cql.patch[CODE]:
            return dependency_h(block)

        generated["lower_h"].append(generated_h)
        return generated_h

    @cql.compiler.qec_lowering(objective=cql.logical.x,
                               codes=(CODE,),
                               dependencies=(dependency_x,))
    def lower_x(site, context):
        if reuse_dependency:
            return dependency_x

        @cql.protocol(implements=cql.logical.x)
        def generated_x(block: cql.patch[CODE]) -> cql.patch[CODE]:
            return dependency_x(block)

        generated["lower_x"].append(generated_x)
        return generated_x

    @cql.program
    def program() -> bool:
        q = cql.prepare_zero()
        q = cql.h(q)
        q = cql.x(q)
        q = cql.h(q)
        q = cql.x(q)
        q = cql.h(q)
        return cql.measure_z(q)

    builder = cql.devices.DeviceBuilder("ManifestDependencies")
    compute = builder.logical.add_compute(capacity=1)
    builder.qec.bind(compute,
                     architecture=cql.devices.QECArchitecture(
                         "manifest_dependencies",
                         CODE.default_encoding,
                         link_roots=(cql.gadgets.prepare_zero(CODE),
                                     cql.gadgets.measure_z(CODE), lower_h,
                                     lower_x)))
    device = builder.build()
    p1 = cql.compiler.place(cql.compile(program), device=device)
    p2 = cql.compile(p1, pipeline=cql.compiler.pipelines.qec(), device=device)
    assert p2.module.operation.verify()
    operations = list(p2.module.body.operations)
    manifests = {
        mlir_ir.StringAttr(op.attributes["sym_name"]).value: op
        for op in operations
        if op.name == "qlx.qec_lowering"
    }
    for name, dependency, count in (("lower_h", "dependency_h", 3),
                                    ("lower_x", "dependency_x", 2)):
        expected = [dependency]
        if not reuse_dependency:
            assert len(generated[name]) == count
            base = name.replace("lower_", "generated_")
            expected.extend([base] + [f"{base}_{i}" for i in range(1, count)])
        actual = [
            mlir_ir.FlatSymbolRefAttr(ref).value
            for ref in manifests[name].attributes["dependencies"]
        ]
        assert actual == expected
    assert cql.compiler.Build.replay(p2.serialize()).to_mlir() == p2.to_mlir()
