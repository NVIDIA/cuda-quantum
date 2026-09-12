# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.programs.definition import DefinitionHandle
from cudaq.logical.gadgets.profiles import GadgetProfile
from cudaq.logical.gadgets.records import (
    ProfileParity,
    RecordVectorParity,
)


class GadgetProfileBuilder:
    """Lower immutable success and boundary analysis to declarative Fabric IR."""

    def __init__(self, transaction, profile: GadgetProfile) -> None:
        self.transaction = transaction
        self.profile = profile
        self.context = transaction.context
        self.location = transaction.location
        self.gadget = transaction.materialize(profile.gadget)
        self.symbol = transaction.unique_symbol(profile.name)
        self._create_profile()

    def _create_profile(self) -> None:
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.symbol, context=self.context),
            "gadget":
                mlir_ir.FlatSymbolRefAttr.get(self.gadget.symbol,
                                              context=self.context),
        }

        def profiles(values):
            entries = []
            for endpoint, profile in values.items():
                handle = self.transaction.materialize(profile)
                entries.append(
                    mlir_ir.DictAttr.get(
                        {
                            "endpoint":
                                mlir_ir.IntegerAttr.get(
                                    mlir_ir.IntegerType.get_signless(
                                        64, context=self.context),
                                    endpoint.index,
                                ),
                            "name":
                                mlir_ir.StringAttr.get(endpoint.name,
                                                       context=self.context),
                            "profile":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    handle.symbol, context=self.context),
                        },
                        context=self.context,
                    ))
            return mlir_ir.ArrayAttr.get(entries, context=self.context)

        attrs["inputs"] = profiles(self.profile.input_profiles)
        attrs["outputs"] = profiles(self.profile.output_profiles)
        if self.profile.boundary_complete:
            attrs["boundary_complete"] = mlir_ir.UnitAttr.get(
                context=self.context)
        if self.profile.metadata:
            attrs["metadata"] = mlir_ir.DictAttr.get(
                {
                    key:
                        mlir_ir.StringAttr.get(str(value), context=self.context)
                    for key, value in self.profile.metadata.items()
                },
                context=self.context,
            )
        with self.location:
            self.operation = mlir_ir.Operation.create(
                "fabric.gadget_profile",
                attributes=attrs,
                regions=1,
                loc=self.location,
            )
            self.transaction.module.body.append(self.operation)
            self.block = self.operation.regions[0].blocks.append()
        self.insertion_point = mlir_ir.InsertionPoint(self.block)

    def _records(self, parity):
        return mlir_ir.ArrayAttr.get(
            [
                mlir_ir.StringAttr.get(f"{self.gadget.symbol}.{record.name}",
                                       context=self.context)
                for record in parity.records
            ],
            context=self.context,
        )

    def _input_syndromes(self, parity):
        return mlir_ir.ArrayAttr.get(
            [
                mlir_ir.DictAttr.get(
                    {
                        "port_index":
                            mlir_ir.IntegerAttr.get(
                                mlir_ir.IntegerType.get_signless(
                                    64, context=self.context),
                                syndrome.endpoint.index,
                            ),
                        "index":
                            mlir_ir.IntegerAttr.get(
                                mlir_ir.IntegerType.get_signless(
                                    64, context=self.context),
                                syndrome.index,
                            ),
                    },
                    context=self.context,
                ) for syndrome in parity.input_syndromes
            ],
            context=self.context,
        )

    def _parity_attrs(self, parity):
        if isinstance(parity, RecordVectorParity):
            raise TypeError(
                "record-vector parities must be normalized to scalar profile rows"
            )
        attrs = {}
        if parity.records:
            attrs["records"] = self._records(parity)
        if isinstance(parity, ProfileParity):
            if parity.input_syndromes:
                attrs["input_syndromes"] = self._input_syndromes(parity)
            if parity.constant:
                attrs["constant"] = mlir_ir.BoolAttr.get(True,
                                                         context=self.context)
        return attrs

    def _common(self, expression, parity=None):
        return self._parity_attrs(expression.parity if parity is
                                  None else parity)

    def _emit(self, name, attributes):
        with self.location:
            operation = mlir_ir.Operation.create(name,
                                                 attributes=attributes,
                                                 loc=self.location)
            self.insertion_point.insert(operation)

    def trace(self):
        for success in self.profile.success:
            attrs = self._common(success)
            self._emit("fabric.success", attrs)
        for assignment in self.profile.output_syndromes:
            attrs = self._parity_attrs(assignment.parity)
            attrs["port_index"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                assignment.endpoint.index,
            )
            attrs["index"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                assignment.index,
            )
            self._emit("fabric.output_syndrome", attrs)
        self._emit("fabric.profile_end", {})

    def handle(self):
        return DefinitionHandle(symbol=self.symbol,
                                kind="gadget_profile",
                                profile="p2a")
