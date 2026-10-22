# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.logical as cql
from cudaq.logical.protocols.builder import ProtocolBuilder


def test_protocol_patch_types_preserve_encoding_epoch_and_context():
    encoding = cql.codes.Steane.default_encoding
    alternate = cql.codes.Encoding(cql.codes.Steane,
                                   name="alternate",
                                   block="alternate_block")
    later = encoding.epoch(encoding.initial_epoch.phase, index=1)

    @cql.protocol
    def identity(block: cql.patch[encoding]) -> cql.patch[encoding]:
        return block

    contexts = [cql.compiler.CompilationContext() for _ in range(2)]
    types = []
    for context in contexts:
        builder = ProtocolBuilder(context, identity)
        initial = builder._patch_type(encoding)
        assert initial == builder._patch_type(encoding,
                                              epoch=encoding.initial_epoch)
        advanced = builder._patch_type(encoding, epoch=later)
        other = builder._patch_type(alternate)
        assert len({str(initial), str(advanced), str(other)}) == 3
        assert later.name in str(advanced)
        assert alternate.name in str(other)
        builder.finish(builder.arguments()[0])
        assert context.module.operation.verify()
        types.append(initial)
    assert str(types[0]) == str(types[1])
    assert types[0] != types[1]


def test_each_protocol_retains_reused_type_dependencies():
    encoding = cql.codes.Steane.default_encoding

    @cql.protocol
    def first(block: cql.patch[encoding]) -> cql.patch[encoding]:
        return block

    @cql.protocol
    def second(block: cql.patch[encoding]) -> cql.patch[encoding]:
        return block

    context = cql.compiler.CompilationContext()
    for definition in (first, second):
        _, dependencies = context.materialize_with_dependencies(definition)
        assert {id(encoding.code),
                id(encoding),
                id(encoding.initial_epoch)
               } <= {id(dependency) for dependency in dependencies}
    assert context.module.operation.verify()
