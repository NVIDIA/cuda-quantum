#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Retain only registered product operations in generated Python bindings."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: filter_generated_ops.py INPUT OUTPUT OP_CLASS...")

    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    operations = frozenset(sys.argv[3:])
    retained = frozenset(
        {"_Dialect", *operations, *(f"{name}Adaptor" for name in operations)})

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    missing = sorted(operations - classes.keys())
    if missing:
        raise SystemExit(
            "registered product operation classes missing from TableGen output: "
            + ", ".join(missing))

    lines = source.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    removed_classes = classes.keys() - retained
    for name in removed_classes:
        node = classes[name]
        decorated = [entry.lineno for entry in node.decorator_list]
        start = min([node.lineno, *decorated]) - 1
        removals.append((start, node.end_lineno))

    # TableGen also emits one snake-case convenience constructor per op. Its
    # function name is not mechanically tied to the Python class name, so use
    # the generated function's typed reference to the removed class as the
    # structural ownership signal.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        referenced = {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        }
        if not referenced.intersection(removed_classes):
            continue
        decorated = [entry.lineno for entry in node.decorator_list]
        start = min([node.lineno, *decorated]) - 1
        removals.append((start, node.end_lineno))

    for start, end in sorted(removals, reverse=True):
        del lines[start:end]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
