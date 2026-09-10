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
import re
from pathlib import Path

import cudaq.mlir.ir as mlir_ir

from .._native import native
from ..programs.definition import DefinitionHandle
from .build import Build, EvidenceRecord
from .context import CompilationContext
from .linearity import verify_linearity
from .pipeline import pipelines

# CUDA-Q owns frontend normalization and aggregate-to-linear conversion. QLX's
# one pre-linear transform expands only statically bounded register traversals
# that block that conversion; clean paper-scale workload loops remain folded.
CUDAQ_PREPARATION_PREFIX = (
    "builtin.module(canonicalize,cc-loop-normalize,symbol-dce,"
    "expand-measurements)")
QLX_REGISTER_TRAVERSAL_PIPELINE = (
    "expand-quake-register-traversals{maximum-iterations=4096 "
    "maximum-generated-operations=1000000}")
CUDAQ_LINEAR_VALUES_PIPELINE = (
    "builtin.module(func.func(expand-control-veqs,factor-quantum-alloc,"
    "cable-rough-in,canonicalize,"
    "memtoreg{classical=false quantum=true},canonicalize,"
    "repair-linear-type,linear-ctrl-form),canonicalize)")

# This is the CUDA-Q compile-target override advertised by Target. Keep the
# import path and the public compile-target configuration identical: splitting
# it changes the allocation normalization order and leaves !quake.ref values
# at the typed Quake-to-P0 boundary.
# Passes bracketing `prepare-for-wireset` are QLX-specific and deliberately live
# here rather than in the shared CUDA-Q pipeline. The decomposition basis is the
# gate set Quake-to-P0 accepts.
CUDAQ_TARGET_PREPARATION_PIPELINE = ",".join((
    "expand-measurements",
    "canonicalize",
    "globalize-array-values",
    "canonicalize",
    "prepare-for-wireset{unroll-only-index-use-loops=true maximum-iterations=2048000}",
    "decomposition{basis=h,s,t,x,y,z,x(1),z(1),x(2),z(2),swap,rx,ry,rz,r1}",
    "canonicalize",
    "func.func(normalize-phase-placement)",
    "func.func(lower-phase)",
    "func.func(expand-control-negations)",
    "canonicalize",
    "cse",
))

# Direct ``import_cudaq`` continues through the QLX-owned register-traversal
# expansion and typed P0 conversion in-process.
CUDAQ_TO_P0_PREPARATION_PIPELINE = " -> ".join((
    CUDAQ_PREPARATION_PREFIX,
    QLX_REGISTER_TRAVERSAL_PIPELINE,
    CUDAQ_LINEAR_VALUES_PIPELINE,
    "prepare-quake-for-qlx",
    "convert-quake-to-qlx",
))


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

    This private seam exists so the closed ``import_cudaq`` / ``import_quake``
    paths can run one authoritative linearity check while constructing their
    immutable :class:`Build`. The public low-level conversion API above remains
    independently checked.
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


def _prepare_cudaq_module(module, pass_manager_type) -> None:
    """Reach the strict scalar-wire boundary in the declared pass order."""

    prefix = pass_manager_type.parse(CUDAQ_PREPARATION_PREFIX,
                                     context=module.context)
    prefix.run(module.operation)
    native.run_pass_capsule(module, QLX_REGISTER_TRAVERSAL_PIPELINE)
    linear = pass_manager_type.parse(CUDAQ_LINEAR_VALUES_PIPELINE,
                                     context=module.context)
    linear.run(module.operation)


def _commitment(value) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception as error:
        raise TypeError(
            "CUDA-Q specialization arguments must have a stable JSON "
            "representation") from error
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_cudaq_source(text: str) -> str:
    """Remove CUDA-Q's process-local symbol uniquers from a source commitment."""
    return re.sub(r"\.\.0x[0-9a-fA-F]+", "..<unique>", text)


def _close_cudaq_decorator_helpers(kernel, module):
    """Merge and synthesize captured ``@cudaq.kernel`` helper definitions.

    CUDA-Q 0.15's public ``synthesize`` specializes ordinary arguments but
    leaves decorator helpers as lifted ``!cc.callable`` arguments. Its runtime
    already owns the exact closure conversion used for execution, so use that
    compatibility surface recursively. The native importer then retains the
    now-direct reachable helper calls as typed ``qlx.call`` operations. Newer
    CUDA-Q releases should make this closure step part of their public
    optimizer-form ingress.
    """

    from cudaq.kernel.kernel_decorator import DecoratorCapture
    from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
    from cudaq.mlir.ir import StringAttr

    def close_helper(decorator):
        helper_module = cudaq_runtime.cloneModule(decorator.qkeModule)
        captures = decorator.resolve_captured_arguments()
        callable_names = []
        for capture in captures:
            if not isinstance(capture, DecoratorCapture):
                continue
            helper_module = cudaq_runtime.mergeExternalMLIR(
                helper_module,
                close_helper(capture.decorator),
            )
            callable_names.append(capture.decorator.uniqName)
        if callable_names:
            cudaq_runtime.synthPyCallable(helper_module, callable_names)
        return helper_module

    callable_names = []
    for capture in kernel.resolve_captured_arguments():
        if not isinstance(capture, DecoratorCapture):
            continue
        module = cudaq_runtime.mergeExternalMLIR(
            module,
            close_helper(capture.decorator),
        )
        callable_names.append(capture.decorator.uniqName)
    if callable_names:
        cudaq_runtime.synthPyCallable(module, callable_names)

    # Helper symbols are public in CUDA-Q's source modules. Making non-entry
    # definitions private lets symbol-dce retain exactly the selected reachable
    # call graph before the strict Quake-to-P0 boundary.
    for operation in module.body:
        if operation.operation.name != "func.func":
            continue
        if "cudaq-entrypoint" not in operation.attributes:
            operation.attributes["sym_visibility"] = StringAttr.get(
                "private", context=module.context)
    return module


def _build_from_p0(
    module,
    *,
    root: str | None,
    origin: str,
    source_sha256: str,
    producer: str,
    assumptions: tuple[str, ...],
    source_modules: tuple[str, ...],
    specialization: dict[str, str] | None = None,
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
    if specialization is not None:
        program.attributes["specialization"] = mlir_ir.DictAttr.get(
            {
                key: mlir_ir.StringAttr.get(value, context=context)
                for key, value in specialization.items()
            },
            context=context,
        )

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
        specialization=None,
    )


def import_cudaq(
        kernel,
        *arguments,
        source_modules: tuple[str, ...] = (),
        _objective: bool = False,
) -> Build:
    """Compile a standard ``@cudaq.kernel`` into a verified QLX P0 Build.

    CUDA-Q owns source compilation, argument specialization, helper closure,
    and aggregate-to-linear conversion. QLX retains reachable specialized
    helper calls as typed P0 ``qlx.call`` operations, expands the normalized,
    statically bounded register traversals that block wire conversion, then
    imports the closed value-semantic module as ordinary machine-free P0.
    Unsupported residual Quake/CC, recursive helpers, or unresolved vector
    specialization fails closed.
    """

    try:
        import cudaq
        from cudaq.mlir.passmanager import PassManager
    except ImportError as error:
        raise RuntimeError(
            "import_cudaq requires the CUDA-Q Python package matching the "
            "QLX CUDA-Q SDK build") from error

    name = getattr(kernel, "name", None)
    if not isinstance(name, str) or not hasattr(kernel, "qkeModule"):
        raise TypeError("kernel must be a standard @cudaq.kernel object")

    try:
        specialized = cudaq.synthesize(kernel, *arguments)
        module = _close_cudaq_decorator_helpers(kernel, specialized.qkeModule)
    except Exception as error:
        raise ValueError(
            f"CUDA-Q could not specialize kernel {name!r} for QLX import"
        ) from error

    if _objective:
        from cudaq.mlir.ir import UnitAttr

        candidates = [
            view for view in module.body.operations
            if view.operation.name == "func.func" and "cudaq-kernel" in
            view.operation.attributes and name in _symbol(view.operation)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"CUDA-Q objective kernel {name!r} did not produce exactly one "
                "matching Quake function")
        function = candidates[0]
        with module.context:
            function.attributes["cudaq-entrypoint"] = UnitAttr.get(
                context=module.context)
        inline = PassManager.parse("builtin.module(inline)",
                                   context=module.context)
        inline.run(module.operation)

    source = _stable_cudaq_source(str(module))
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    argument_commitment = _commitment(arguments)
    arguments_json = json.dumps(arguments,
                                sort_keys=True,
                                separators=(",", ":"))
    try:
        native.load_plugin(_quake_plugin_path())
        _prepare_cudaq_module(module, PassManager)
    except Exception as error:
        raise ValueError(
            f"Quake preparation failed for kernel {name!r}; QLX requires a "
            "closed, specialized, scalar !quake.wire module") from error

    cudaq_version = str(getattr(cudaq, "__version__", "unknown"))
    return _build_from_p0(
        _run_quake_to_p0_pass(module, root=name),
        root=name,
        origin=f"@cudaq.kernel:{name}",
        source_sha256=source_sha256,
        producer="cudaq-logical-cudaq-import@0.1",
        assumptions=(
            f"cudaq-version:{cudaq_version}",
            "source-normalization:cudaq-unique-symbol-suffix",
            f"preparation:{CUDAQ_TO_P0_PREPARATION_PIPELINE}",
            f"specialization:{argument_commitment}",
        ),
        source_modules=source_modules,
        specialization={
            "cudaq_entry": name,
            "cudaq_arguments_json": arguments_json,
            "cudaq_arguments_sha256": argument_commitment,
        },
    )
