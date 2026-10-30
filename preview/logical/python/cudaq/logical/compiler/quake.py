# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native CUDA-Q Quake to canonical QLX P0 import."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cudaq.mlir.ir as mlir_ir

from .._native import native
from ..programs.definition import DefinitionHandle
from .build import Build, EvidenceRecord
from .context import CompilationContext
from .linearity import verify_linearity
from .pipeline import pipelines


def _text(source) -> tuple[str, str]:
    if isinstance(source, Path):
        return source.read_text(), str(source)
    if not isinstance(source, str):
        raise TypeError("Quake source must be MLIR text or a pathlib.Path")
    # Strings are always source text. Requiring Path for files avoids guessing
    # whether a short MLIR fragment happens to match a filesystem entry.
    return source, "<quake>"


def _symbol(operation) -> str:
    attribute = operation.attributes["sym_name"]
    return str(getattr(attribute, "value", attribute)).strip('"')


def _quake_plugin_path() -> str:
    """Locate the loadable Quake-import plugin (.so/.dylib).

    Honors the ``QLX_QUAKE_PLUGIN_LIB`` override, then looks in the ``_quake``
    directory of the qlx package (populated by the ``qlx-quake-plugin`` build).
    """
    override = os.environ.get("QLX_QUAKE_PLUGIN_LIB", "")
    if override and os.path.isfile(override):
        return override
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_quake")
    for name in (
            "libqlx-quake-plugin.dylib",
            "libqlx-quake-plugin.so",
            "qlx-quake-plugin.dylib",
            "qlx-quake-plugin.so",
    ):
        candidate = os.path.join(pkg_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "QLX Quake-import plugin not found. Checked $QLX_QUAKE_PLUGIN_LIB and "
        f"{pkg_dir}; build the qlx-quake-plugin target.")


def _convert_to_p0(text: str, *, root: str | None = None) -> mlir_ir.Module:
    """Convert value-semantic Quake text to a P0 module, in-process.

    Loads the Quake-import plugin, which registers the quake/cc dialects backed
    by CUDA-Q's shared MLIR library, parses the Quake, and runs the typed
    ``convert-quake-to-qlx`` pass owned by the QLX host runtime -- all
    in-process, with no subprocess and no output-text round-trip.

    The P0 boundary is enforced by the QLX op verifiers (``ProgramOp`` /
    ``ApplyOp`` / ``PrepareOp``), which recursively check the program body for the
    structural P0 invariants: a machine-free single-block program, a matching
    signature, only P0 dialects, and no region values. We assert them explicitly
    on the live module with ``module.operation.verify()``. Linear ownership is
    checked separately by ``verify_linearity`` during :class:`Build` construction.
    """
    if not native.has_quake_import:
        raise RuntimeError(
            "QLX was built without typed Quake import support; enable "
            "QLX_USE_CUDAQ_SDK")
    # Idempotently registers the shared CUDA-Q Quake/CC dialects. The typed
    # conversion pass itself is registered by the QLX host runtime.
    native.load_plugin(_quake_plugin_path())
    context = mlir_ir.Context()
    module = mlir_ir.Module.parse(text, context)
    return _run_quake_to_p0_pass(module, root=root)


def convert_quake_to_p0(module, *, root: str | None = None):
    """Run the typed Quake-to-P0 pass on a live MLIR module.

    The module is transactionally replaced and returned; there is no file,
    subprocess, or text round-trip. It may be a QLX MLIR module or another
    MLIR Python module that
    exposes the standard C-API module handle. In a ``QLX_USE_CUDAQ_SDK`` build,
    CUDA-Q and QLX resolve Quake/CC through the same shared compiler library,
    so typed operation dispatch uses the same dialect TypeIDs. The packaged
    ``qlx-quake-plugin`` remains a runtime dependency: it is loaded here to
    register those shared Quake/CC dialects, while the typed conversion pass
    itself lives in the QLX host runtime.

    This is the low-level pass surface. Use :func:`import_quake` when a frozen,
    verified P0 :class:`Build` and import evidence are desired.
    """
    converted = _run_quake_to_p0_pass(module, root=root)
    verify_linearity(converted, subject="converted Quake P0 module")
    # The closed import path intentionally transfers the converted P0 into a
    # fresh QLX context so all later-stage dialects are registered.  This
    # low-level API instead promises to mutate the caller-owned module.  Move
    # the verified result back through typed MLIR bytecode before the atomic
    # body replacement; the original context already hosted the conversion
    # pass and therefore knows the emitted QLX dialect.
    if converted.context is not module.context:
        converted = mlir_ir.Module._CAPICreate(
            native.clone_module_into_context_capsule(converted, module.context))
    native.replace_module_contents_capsule(module, converted)
    return module


def _run_quake_to_p0_pass(module, *, root: str | None = None):
    """Convert a live module, deferring linearity to the owning caller.

    ``import_quake`` checks linearity once, while it builds the immutable
    :class:`Build`. :func:`convert_quake_to_p0` checks linearity itself.
    """

    if not hasattr(module, "operation"):
        raise TypeError("module must be a live MLIR module")
    if not native.has_quake_import:
        raise RuntimeError(
            "QLX was built without typed Quake import support; enable "
            "QLX_USE_CUDAQ_SDK")
    native.load_plugin(_quake_plugin_path())
    converted = mlir_ir.Module._CAPICreate(native.clone_module_capsule(module))
    pipeline = "prepare-quake-for-qlx,convert-quake-to-qlx"
    if root is not None:
        if not isinstance(root, str) or not root:
            raise TypeError("root must be a non-empty CUDA-Q entry-point name")
        pipeline += "{entry-point=" + json.dumps(root) + "}"
    native.run_pass_capsule(converted, pipeline)
    # CUDA-Q owns the Quake module's MLIRContext, whose registry contains only
    # the dialects required by its frontend and the P0 conversion pass.  P0 is
    # an extensible compiler artifact: later placement, QEC selection, and
    # physical projection must materialize LVM, Fabric, and Phys operations in
    # the same live module.  Transfer the converted module by typed MLIR
    # bytecode into a fresh QLX context, which snapshots QLX's complete dialect
    # registry.  This is a binary IR transfer, not textual emission/reparsing.
    qlx_context = mlir_ir.Context()
    converted = mlir_ir.Module._CAPICreate(
        native.clone_module_into_context_capsule(converted, qlx_context))
    if not converted.operation.verify():
        raise RuntimeError(
            "convert-quake-to-qlx produced a module that failed QLX P0 "
            "op verification")
    return converted


def _build_from_p0(
    module,
    *,
    root: str | None,
    origin: str,
    source_sha256: str,
    producer: str,
    assumptions: tuple[str, ...],
    source_modules: tuple[str, ...],
) -> Build:
    if not isinstance(module, mlir_ir.Module):
        if getattr(module, "_CAPIPtr", None) is None:
            raise TypeError("converted module does not expose the MLIR C API")
        module = mlir_ir.Module._CAPICreate(native.clone_module_capsule(module))
    transaction = CompilationContext(module=module)
    context = transaction.context
    module = transaction.module

    # Mirror the P0 profile into qlx.profiles/qlx.stages/qlx.facets exactly as
    # the native @cudaq.logical.program builder does (builders/portable.py).
    transaction.add_profile("p0")

    programs = {
        _symbol(view.operation): view.operation
        for view in module.body.operations
        if view.operation.name == "qlx.program"
    }
    if root is None:
        if len(programs) != 1:
            raise ValueError(
                "root= is required when Quake contains more than one "
                "CUDA-Q entry point")
        root = next(iter(programs))
    root = root.removeprefix("__nvqpp__mlirgen__")
    if ".." in root:
        root = root.split("..", 1)[0]
    if root not in programs:
        raise ValueError(f"Quake entry point {root!r} was not imported")

    program = programs[root]
    program.attributes["qlx.source_sha256"] = mlir_ir.StringAttr.get(
        source_sha256, context=context)
    program.attributes["qlx.source_origin"] = mlir_ir.StringAttr.get(
        origin, context=context)

    value_groups: dict[str, int] = {}

    def nested_operations(operation):
        for region in operation.regions:
            for block in region.blocks:
                for view in block.operations:
                    child = view.operation
                    yield child
                    yield from nested_operations(child)

    for operation in nested_operations(program):
        if operation.name != "qlx.prepare":
            continue
        allocation = int(operation.attributes["allocation"])
        key = f"alloc{allocation}"
        value_groups[key] = value_groups.get(key, 0) + 1

    report = verify_linearity(module, subject=f"imported Quake program @{root}")
    return Build(
        context=context,
        module=module,
        root=DefinitionHandle(root, "program", "p0"),
        profile="p0",
        pipeline=pipelines.logical(),
        evidence=(
            EvidenceRecord(
                kind="source_import",
                producer=producer,
                result="pass",
                obligations=("quake-wire-semantics", "p0-refinement"),
                assumptions=(
                    f"source:{origin}",
                    f"source-sha256:{source_sha256}",
                    *assumptions,
                ),
            ),
            EvidenceRecord(
                kind="linearity_verification",
                producer="cudaq-logical-python@0.3",
                result=report.result,
                obligations=("linear-ownership",),
                assumptions=(
                    "checker:qlx-linear-use/v1",
                    f"checked-bodies:{len(report.checked_bodies)}",
                    f"linear-values:{report.linear_values}",
                ),
            ),
        ),
        value_groups=value_groups,
        source_modules=source_modules,
    )


def import_quake(
        source,
        *,
        root: str | None = None,
        source_modules: tuple[str, ...] = (),
) -> Build:
    """Import ``!quake.wire`` CUDA-Q Quake as a verified P0 ``Build``.

    ``source`` may be Quake MLIR text or a :class:`pathlib.Path`.
    ``root`` selects the source entry before conversion and is required when
    the source contains more than one CUDA-Q entry point. An abbreviated name
    is accepted only when it identifies exactly one source entry.
    ``source_modules`` explicitly links Python modules containing QEC gadgets
    needed by later P2 lowering; these links do not alter P0 conversion. The
    native pass accepts only linear value-semantics ``!quake.wire`` quantum
    values; this wrapper performs no Quake normalization or lowering.
    """

    text, origin = _text(source)
    source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _build_from_p0(
        _convert_to_p0(text, root=root),
        root=root,
        origin=origin,
        source_sha256=source_sha256,
        producer="cudaq-logical-quake-import@0.2",
        assumptions=("preparation:pre-normalized",),
        source_modules=source_modules,
    )


def import_cudaq(
        kernel,
        *arguments,
        target=None,
        source_modules: tuple[str, ...] = (),
) -> Build:
    """Compile a ``@cudaq.kernel`` for a Logical target and import it as P0.

    CUDA-Q compiles the kernel with ``target``'s compile target and
    substitutes ``arguments``, as a launch on that target would.
    :func:`import_quake` then imports the result. ``target`` is the Logical
    target that consumes the build. It defaults to
    ``cudaq.logical.targets.estimator``, never to the target installed with
    ``cudaq.set_target``.
    """

    from cudaq.kernel.kernel_decorator import isa_kernel_decorator
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime

    if not isa_kernel_decorator(kernel):
        raise TypeError("kernel must be a standard @cudaq.kernel object")
    if target is None:
        from ..targets import estimator as target
    # Take the endpoint's traits from ``target``, never from the installed
    # target, so the compile is never emulated. Logical's endpoint reads the
    # MLIR, so it declares no JIT support and the compile builds no JIT engine.
    endpoint = target.runtime_endpoint
    processed, module = kernel.prepare_call(*arguments)
    compiled = cudaq_runtime.compile_module(
        kernel.uniqName,
        module,
        is_entry_point=True,
        compile_target=target.compile_target,
        is_remote=getattr(endpoint, "is_remote", False),
        is_emulated=getattr(endpoint, "is_emulated", False),
        supports_jit=getattr(endpoint, "supports_jit", True),
        args=tuple(processed),
    )
    return import_quake(str(compiled.mlir_module),
                        root=kernel.name,
                        source_modules=source_modules)
