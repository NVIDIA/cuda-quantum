# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PulseIREmitter -- walks a RegionTree and emits pulse IR.

Converts from stack-based bytecode semantics to SSA-style pulse IR
using the existing PythonIRBuilder. Resolves Python globals/builtins
at emit time so that expressions like ``int(4 * sigma)`` and
``math.pi / 2`` evaluate to concrete values, while pulse ops
(``drive``, ``gaussian``, etc.) are emitted as IR operations.
"""

from __future__ import annotations

import builtins as _builtins_mod
import operator as pyop
from typing import Any, Union

from .ir_builder import (
    CompilationError,
    IRValue,
    LINEAR_TYPES,
    OP_TABLE,
    Parameter,
    PythonIRBuilder,
)
from ._bytecode_normalize import CanonicalInstr
from ._structure import Block, ForLoop, IfElse, Region

_ALLOC_OPS = {"qudit_ref": "pulse.qudit_alloc", "qvec_ref": "pulse.qvec_alloc"}
_PULSE_OP_NAMES = frozenset(OP_TABLE.keys()) | frozenset(_ALLOC_OPS.keys())

_BIN_OP_FN: dict[str, Any] = {
    "+": pyop.add,
    "-": pyop.sub,
    "*": pyop.mul,
    "/": pyop.truediv,
    "//": pyop.floordiv,
    "%": pyop.mod,
    "**": pyop.pow,
}

StackVal = Union[IRValue, Any]


def _loop_var_used(name: str, regions: list[Region]) -> bool:
    """Check if a loop variable is referenced anywhere in the body regions."""
    for region in regions:
        if isinstance(region, Block):
            for ci in region.instrs:
                if ci.op == "LOAD_FAST" and ci.arg == name:
                    return True
        elif isinstance(region, ForLoop):
            if _loop_var_used(name, region.body):
                return True
        elif isinstance(region, IfElse):
            if _loop_var_used(name, region.true_body):
                return True
            if _loop_var_used(name, region.false_body):
                return True
    return False


class PulseIREmitter:
    """Walks a region tree and emits pulse IR via PythonIRBuilder."""

    def __init__(
        self,
        builder: PythonIRBuilder,
        fn_globals: dict[str, Any] | None = None,
        fn_closures: dict[str, Any] | None = None,
    ):
        self.b = builder
        self.stack: list[StackVal] = []
        self.locals: dict[str, StackVal] = {}
        self._fn_globals = fn_globals or {}
        self._fn_closures = fn_closures or {}
        self._builtins = vars(_builtins_mod)

    # ── Public entry point ───────────────────────────────────────────

    def emit_regions(self, regions: list[Region]) -> None:
        for region in regions:
            self._emit_region(region)

    # ── Region dispatch ──────────────────────────────────────────────

    def _emit_region(self, region: Region) -> None:
        if isinstance(region, Block):
            self._emit_block(region)
        elif isinstance(region, ForLoop):
            self._emit_for_loop(region)
        elif isinstance(region, IfElse):
            self._emit_if_else(region)
        else:
            raise CompilationError(f"unknown region type: {type(region)}")

    def _emit_block(self, block: Block) -> None:
        for ci in block.instrs:
            self._exec_instr(ci)

    def _emit_for_loop(self, loop: ForLoop) -> None:
        count = self._extract_range_count(loop)
        if not isinstance(count, int):
            raise CompilationError(
                f"loop bound must be compile-time int, got {type(count).__name__}"
            )

        if _loop_var_used(loop.loop_var, loop.body):
            self._unroll_loop(loop, count)
        else:
            self._emit_scf_for(loop, count)

    def _unroll_loop(self, loop: ForLoop, count: int) -> None:
        """Unroll a loop whose induction variable is used in the body."""
        for i in range(count):
            self.locals[loop.loop_var] = i
            self.emit_regions(loop.body)

    def _emit_scf_for(self, loop: ForLoop, count: int) -> None:
        """Emit a structured scf.for loop (induction variable not used)."""
        snap = self._ir_snapshot()

        self.b.emit("scf.for", (), (), {
            "lb": 0,
            "ub": count,
            "step": 1,
            "var": loop.loop_var,
        })

        self.emit_regions(loop.body)

        delta = self._ir_delta(snap)
        if delta:
            names = sorted(delta)
            vals = tuple(delta[n] for n in names)
            self.b.emit("scf.yield", vals, ())
            results = self.b.emit("scf.for_end", (),
                                  tuple(v.vtype for v in vals),
                                  {"iter_args": names})
            for n, r in zip(names, results):
                self.locals[n] = r
        else:
            self.b.emit("scf.for_end", (), ())

    def _emit_if_else(self, ifelse: IfElse) -> None:
        cond = self.stack.pop() if self.stack else None

        if isinstance(cond, IRValue):
            if cond.vtype != "measurement":
                raise CompilationError(
                    f"non-determinable branch on IR value of type {cond.vtype}")
            self._emit_scf_if(cond, ifelse)
        elif cond:
            self.emit_regions(ifelse.true_body)
        else:
            self.emit_regions(ifelse.false_body)

    def _emit_scf_if(self, cond: IRValue, ifelse: IfElse) -> None:
        snap_full = dict(self.locals)
        snap_ir = self._ir_snapshot()

        self.b.emit("scf.if", (cond,), ())
        self.emit_regions(ifelse.true_body)
        true_delta = self._ir_delta(snap_ir)

        self.locals = dict(snap_full)
        if ifelse.false_body:
            self.b.emit("scf.else", (), ())
            self.emit_regions(ifelse.false_body)
        false_delta = self._ir_delta(snap_ir)

        all_names = sorted(set(true_delta) | set(false_delta))
        if all_names:
            vtypes = tuple(
                (true_delta.get(n) or false_delta[n]).vtype for n in all_names)
            results = self.b.emit("scf.if_end", (), vtypes,
                                  {"result_names": all_names})
            for n, r in zip(all_names, results):
                self.locals[n] = r
        else:
            self.b.emit("scf.if_end", (), ())

    # ── Instruction execution ────────────────────────────────────────

    def _exec_instr(self, ci: CanonicalInstr) -> None:
        op = ci.op
        arg = ci.arg

        if op == "LOAD_FAST":
            if arg not in self.locals:
                raise CompilationError(f"undefined local: {arg}")
            self.stack.append(self.locals[arg])

        elif op == "STORE_FAST":
            val = self.stack.pop()
            self.locals[arg] = val

        elif op == "LOAD_CONST":
            self.stack.append(arg)

        elif op == "LOAD_GLOBAL":
            self.stack.append(self._resolve_global(arg))

        elif op == "LOAD_ATTR":
            obj = self.stack.pop()
            if isinstance(
                    obj,
                    tuple) and len(obj) == 2 and obj[0] == "__unresolved__":
                self.stack.append(("__unresolved__", arg))
            else:
                try:
                    self.stack.append(getattr(obj, arg))
                except (AttributeError, TypeError):
                    raise CompilationError(
                        f"cannot resolve attribute {arg!r} on {type(obj).__name__}"
                    )

        elif op == "CALL":
            nargs = arg if isinstance(arg, int) else 0
            call_args = []
            for _ in range(nargs):
                call_args.append(self.stack.pop())
            call_args.reverse()

            _callable = self.stack.pop()
            self._dispatch_call(_callable, call_args)

        elif op == "UNPACK_SEQUENCE":
            val = self.stack.pop()
            if isinstance(val, tuple):
                if len(val) != arg:
                    raise CompilationError(
                        f"unpack mismatch: expected {arg}, got {len(val)}")
                for v in reversed(val):
                    self.stack.append(v)
            elif isinstance(val, list):
                if len(val) != arg:
                    raise CompilationError(
                        f"unpack mismatch: expected {arg}, got {len(val)}")
                for v in reversed(val):
                    self.stack.append(v)
            else:
                raise CompilationError(f"cannot unpack {type(val)}")

        elif op == "POP_TOP":
            if self.stack:
                self.stack.pop()

        elif op == "BINARY_OP":
            right = self.stack.pop()
            left = self.stack.pop()
            fn = _BIN_OP_FN.get(arg)
            if fn is None:
                raise CompilationError(f"unsupported binary op: {arg}")
            self.stack.append(fn(left, right))

        elif op == "UNARY_NEGATIVE":
            val = self.stack.pop()
            self.stack.append(-val)

        elif op == "COMPARE_OP":
            right = self.stack.pop()
            left = self.stack.pop()
            cmp_ops = {
                "<": pyop.lt,
                "<=": pyop.le,
                "==": pyop.eq,
                "!=": pyop.ne,
                ">": pyop.gt,
                ">=": pyop.ge,
            }
            fn = cmp_ops.get(arg)
            if fn is None:
                raise CompilationError(f"unsupported comparison: {arg}")
            self.stack.append(fn(left, right))

        elif op == "BUILD_TUPLE":
            items = []
            for _ in range(arg):
                items.append(self.stack.pop())
            items.reverse()
            self.stack.append(tuple(items))

        elif op == "BUILD_LIST":
            items = []
            for _ in range(arg):
                items.append(self.stack.pop())
            items.reverse()
            self.stack.append(items)

        elif op == "LIST_EXTEND":
            iterable = self.stack.pop()
            lst = self.stack[-1]
            if isinstance(lst, list):
                lst.extend(iterable)
            else:
                raise CompilationError(f"LIST_EXTEND on non-list: {type(lst)}")

        elif op == "LIST_APPEND":
            val = self.stack.pop()
            lst = self.stack[-1]
            if isinstance(lst, list):
                lst.append(val)
            else:
                raise CompilationError(f"LIST_APPEND on non-list: {type(lst)}")

        elif op == "BINARY_SUBSCR":
            index = self.stack.pop()
            obj = self.stack.pop()
            try:
                self.stack.append(obj[index])
            except (TypeError, IndexError, KeyError) as e:
                raise CompilationError(
                    f"subscript failed: {type(obj).__name__}[{index!r}]: {e}")

        elif op == "GET_ITER":
            pass  # handled in for-loop structure recovery

        elif op == "FOR_ITER":
            pass  # handled in structure recovery

        elif op == "JUMP":
            pass  # unconditional jumps are consumed by structure recovery

        elif op == "RETURN":
            pass  # function return at end of kernel

        elif op == "IMPORT_NAME":
            self.stack.append(self._resolve_global(arg))

        elif op in ("LOAD_DEREF", "LOAD_CLOSURE"):
            if arg in self._fn_closures:
                self.stack.append(self._fn_closures[arg])
            else:
                self.stack.append(self._resolve_global(arg))

        elif op == "STORE_DEREF":
            val = self.stack.pop()
            self.locals[arg] = val

        elif op == "JUMP_IF_FALSE" or op == "JUMP_IF_TRUE":
            pass  # consumed by structure recovery for if/else

        elif op == "JUMP_BACKWARD":
            raise CompilationError(
                "unsupported control flow: while loops and break/continue "
                "are not supported in @cudaq_pulse.kernel; use for loops")

        elif op == "SWAP":
            n = arg if isinstance(arg, int) else 2
            if n == 2 and len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
            elif n == 3 and len(self.stack) >= 3:
                self.stack[-1], self.stack[-3] = self.stack[-3], self.stack[-1]
            elif len(self.stack) >= n:
                self.stack[-1], self.stack[-n] = self.stack[-n], self.stack[-1]

        elif op in ("ROT_TWO", "ROT_THREE", "DUP_TOP"):
            pass

        elif op == "BUILD_CONST_KEY_MAP":
            keys = self.stack.pop()
            vals = []
            for _ in range(arg):
                vals.append(self.stack.pop())
            vals.reverse()
            self.stack.append(dict(zip(keys, vals)))

        elif op == "BUILD_MAP":
            d: dict = {}
            for _ in range(arg):
                v = self.stack.pop()
                k = self.stack.pop()
                d[k] = v
            self.stack.append(d)

        elif op == "STORE_SUBSCR":
            index = self.stack.pop()
            obj = self.stack.pop()
            value = self.stack.pop()
            obj[index] = value

        else:
            raise CompilationError(
                f"unsupported bytecode instruction: {op} (arg={arg})")

    # ── Global / closure resolution ──────────────────────────────────

    def _resolve_global(self, name: str) -> Any:
        """Resolve a global name to its Python value."""
        if name in self._fn_globals:
            return self._fn_globals[name]
        if name in self._fn_closures:
            return self._fn_closures[name]
        if name in self._builtins:
            return self._builtins[name]
        return ("__unresolved__", name)

    # ── Call dispatch ────────────────────────────────────────────────

    def _dispatch_call(self, callable_val: Any,
                       call_args: list[StackVal]) -> None:
        """Route a call to pulse IR emission or Python evaluation."""

        # Unresolved sentinel (fallback from _resolve_global)
        if isinstance(callable_val, tuple) and len(callable_val) == 2:
            tag, name = callable_val
            if tag == "__unresolved__":
                if name in OP_TABLE or name in _ALLOC_OPS:
                    self._push_pulse_results(name, call_args)
                elif name == "range":
                    self.stack.append(("__range__", call_args))
                else:
                    raise CompilationError(f"unknown pulse op: {name}")
                return

        # Resolved Python callable
        fname = getattr(callable_val, "__name__", "")

        # Qudit/qvec allocation (check before OP_TABLE so qvec_ref gets expanded)
        if fname in _ALLOC_OPS:
            self._push_alloc_results(fname, call_args)
            return

        # Pulse ops
        if fname in OP_TABLE:
            self._push_pulse_results(fname, call_args)
            return

        # range() -> sentinel for for-loop handling
        if callable_val is range or fname == "range":
            self.stack.append(("__range__", call_args))
            return

        # Regular Python callable (int, float, len, abs, math.sin, etc.)
        # If any arg is a Parameter, we can't evaluate -- propagate the param
        if callable(callable_val):
            if any(isinstance(a, Parameter) for a in call_args):
                raise CompilationError(
                    f"Cannot call {fname}() with Parameter arguments. "
                    f"Parameters must be passed directly to pulse ops "
                    f"(gaussian, drive, shift_phase, etc.), not through "
                    f"Python functions like {fname}().")
            try:
                result = callable_val(*call_args)
            except Exception as e:
                raise CompilationError(
                    f"failed to evaluate {fname}({call_args!r}): {e}")
            self.stack.append(result)
            return

        raise CompilationError(f"cannot call: {callable_val!r}")

    def _push_pulse_results(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> None:
        results = self._emit_pulse_call(fname, call_args)
        if len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)

    def _push_alloc_results(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> None:
        """Handle qudit_ref() and qvec_ref(n) allocation inside kernels."""
        if fname == "qvec_ref":
            if len(call_args) == 1 and isinstance(call_args[0], int):
                n = call_args[0]
                ir_vals = []
                for i in range(n):
                    (v,) = self.b.emit("pulse.qudit_alloc", (), ("qref",),
                                       {"index": i})
                    ir_vals.append(v)
                self.stack.append(ir_vals)
                return
        # qudit_ref() or fallback qvec_ref
        results = self._emit_pulse_call(fname, call_args)
        if len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)

    # ── Pulse IR emission ────────────────────────────────────────────

    def _emit_pulse_call(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> tuple[IRValue, ...]:
        entry = OP_TABLE.get(fname)
        if entry is None:
            raise CompilationError(f"unknown pulse op: {fname}")

        n_val, attr_names, rtypes = entry

        if n_val == -1:
            operands = tuple(v for v in call_args if isinstance(v, IRValue))
            attrs: dict[str, Any] = {}
            out_types = tuple(
                v.vtype for v in operands) if rtypes is None else rtypes
        else:
            operands = tuple(call_args[:n_val])
            attr_vals = call_args[n_val:]
            if len(attr_vals) != len(attr_names):
                raise CompilationError(
                    f"{fname}: expected {n_val + len(attr_names)} args, "
                    f"got {len(call_args)}")
            attrs = dict(zip(attr_names, attr_vals))
            out_types = ((operands[0].vtype,) if
                         (rtypes is None and operands) else (rtypes or ()))

        op_name = _ALLOC_OPS.get(fname, f"pulse.{fname}")
        results = self.b.emit(op_name, operands, out_types, attrs)

        self._rebind_linear(fname, call_args, results)

        return results

    def _rebind_linear(
        self,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """Rebind local names for linear-typed results (drive_line, tone, etc)."""
        arg_info: list[tuple[str | None, IRValue | None]] = []
        for a in call_args:
            if isinstance(a, IRValue):
                name = self._find_local_name(a)
                arg_info.append((name, a))
            else:
                arg_info.append((None, None))

        claimed: set[int] = set()
        for res in results:
            if res.vtype not in LINEAR_TYPES:
                continue
            for i, (name, op_val) in enumerate(arg_info):
                if i in claimed or op_val is None or name is None:
                    continue
                if op_val.vtype == res.vtype:
                    self.locals[name] = res
                    claimed.add(i)
                    break

    def _find_local_name(self, val: IRValue) -> str | None:
        """Find the local variable name bound to a given IRValue."""
        for name, v in self.locals.items():
            if v is val:
                return name
        return None

    # ── Range extraction ─────────────────────────────────────────────

    def _extract_range_count(self, loop: ForLoop) -> Any:
        """Extract the range count, checking the stack first then setup instrs."""
        if self.stack:
            top = self.stack[-1]
            if isinstance(top,
                          tuple) and len(top) == 2 and top[0] == "__range__":
                self.stack.pop()
                args = top[1]
                if len(args) == 1:
                    return args[0]
                elif len(args) == 2:
                    start, stop = args
                    if isinstance(start, int) and isinstance(stop, int):
                        return stop - start
                    raise CompilationError(
                        f"range(start, stop) requires compile-time int args, "
                        f"got range({type(start).__name__}, {type(stop).__name__})"
                    )
                elif len(args) == 3:
                    start, stop, step = args
                    if (isinstance(start, int) and isinstance(stop, int) and
                            isinstance(step, int) and step != 0):
                        return max(0, (stop - start + step - 1) // step) if step > 0 \
                            else max(0, (start - stop - step - 1) // (-step))
                    raise CompilationError(
                        f"range(start, stop, step) requires compile-time int args"
                    )

        for ci in loop.range_setup:
            if ci.op == "LOAD_CONST":
                return ci.arg
        for ci in loop.range_setup:
            if ci.op == "LOAD_FAST" and ci.arg in self.locals:
                return self.locals[ci.arg]
        return None

    # ── Snapshot/delta for structured control flow ───────────────────

    def _ir_snapshot(self) -> dict[str, IRValue]:
        return {k: v for k, v in self.locals.items() if isinstance(v, IRValue)}

    def _ir_delta(self, snap: dict[str, IRValue]) -> dict[str, IRValue]:
        return {
            k: v
            for k, v in self.locals.items()
            if isinstance(v, IRValue) and (k not in snap or snap[k] is not v)
        }
