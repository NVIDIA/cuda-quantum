# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Functions to convert OpenQASM files to CUDA-Q kernels.

This module provides native OpenQASM 2.0 and 3.0 parser implementations
with no Qiskit dependency. `from_qasm_str` dispatches on the
`OPENQASM <version>;` header:
version `2.x` is handled by `_QASM2Translator`, version `3.x` by the
`_QASM3Translator` subclass.

Public API:
    - `from_qasm(path)`          — parse a file from disk.
    - `from_qasm_str(source)`    — parse raw OpenQASM source.
"""

import ast
import math
import operator
import re

from ..kernel.kernel_builder import make_kernel
from .qiskit_convert import _GATE_HANDLERS

_ALLOWED_FUNCS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'exp': math.exp,
    'ln': math.log,
    'log': math.log,
    'sqrt': math.sqrt,
}

_ALLOWED_CONSTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expr(expr_str, env=None):
    """Evaluate a QASM numeric expression using a restricted AST walker.

    Supports: numeric literals, `pi`/`e`/`tau`, `sin`/`cos`/`tan`/`asin`/
    `acos`/`atan`/`exp`/`ln`/`log`/`sqrt`, +-*/**^, and unary +/-.
    Local parameter names (from gate definitions) resolve via `env`.
    """
    env = env or {}
    tree = ast.parse(expr_str.strip(), mode='eval')
    return _eval_node(tree.body, env)


def _eval_node(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Constant):
        return float(node.n)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"Unknown identifier in expression: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(
                f"Unsupported binary operator: {type(node.op).__name__}")
        return op(_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(
                f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand, env))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Only direct function calls are allowed in expressions")
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(
                f"Unknown function in expression: {node.func.id!r}")
        return fn(*[_eval_node(a, env) for a in node.args])
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _strip_comments(source):
    """Remove `// line` and `/* block */` comments."""
    source = re.sub(r'/\*[\s\S]*?\*/', ' ', source)
    source = re.sub(r'//[^\n]*', '', source)
    return source


def _split_top_level(s, sep):
    """Split `s` at `sep` characters not enclosed in (), [], or {}."""
    out, current, depth = [], [], 0
    for c in s:
        if c in '([{':
            depth += 1
            current.append(c)
        elif c in ')]}':
            depth -= 1
            current.append(c)
        elif c == sep and depth == 0:
            out.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        out.append(''.join(current).strip())
    return out


def _parse_op_statement(stmt):
    """Parse `NAME[(PARAMS)] Q0, Q1, ...` → `(name, param_exprs, qubit_refs)`."""
    stmt = stmt.strip()
    m = re.match(r'([A-Za-z_]\w*)', stmt)
    if not m:
        raise ValueError(f"Cannot parse op statement: {stmt!r}")
    name = m.group(1)
    rest = stmt[m.end():].lstrip()

    params_str = ''
    if rest.startswith('('):
        depth, i = 1, 1
        while i < len(rest) and depth > 0:
            if rest[i] == '(':
                depth += 1
            elif rest[i] == ')':
                depth -= 1
            i += 1
        if depth != 0:
            raise ValueError(f"Unmatched '(' in: {stmt!r}")
        params_str = rest[1:i - 1]
        rest = rest[i:].lstrip()

    params = [p for p in _split_top_level(params_str, ',') if p]
    qs = [q for q in _split_top_level(rest, ',') if q]
    return name, params, qs


def _extract_gate_defs(source):
    """Pull out `gate ... { ... }` definitions; return `(stripped_source, definitions)`."""
    gate_defs = {}
    out = []
    i = 0
    n = len(source)
    kw = re.compile(r'\bgate\b')
    while i < n:
        m = kw.search(source, i)
        if not m:
            out.append(source[i:])
            break
        out.append(source[i:m.start()])
        # Find '{'
        brace = source.find('{', m.end())
        if brace < 0:
            raise ValueError("Gate definition missing '{'")
        header = source[m.end():brace].strip()
        # Find matching '}'
        depth, j = 1, brace + 1
        while j < n and depth > 0:
            if source[j] == '{':
                depth += 1
            elif source[j] == '}':
                depth -= 1
            j += 1
        if depth != 0:
            raise ValueError("Unmatched '{' in gate definition")
        body_src = source[brace + 1:j - 1]

        # Header: NAME ['(' PARAMS ')'] ARG_LIST
        hm = re.match(r'([A-Za-z_]\w*)', header)
        if not hm:
            raise ValueError(f"Cannot parse gate name: {header!r}")
        gname = hm.group(1)
        rest = header[hm.end():].lstrip()
        params = []
        if rest.startswith('('):
            end = rest.index(')')
            params = [p for p in _split_top_level(rest[1:end], ',') if p]
            rest = rest[end + 1:].lstrip()
        args = [a for a in _split_top_level(rest, ',') if a]

        body_stmts = []
        for raw in _split_top_level(body_src, ';'):
            raw = raw.strip()
            if not raw:
                continue
            body_stmts.append(_parse_op_statement(raw))
        gate_defs[gname] = {
            'params': params,
            'args': args,
            'body': body_stmts,
        }
        i = j
    return ''.join(out), gate_defs


class _QASM2Translator:
    """Parses OpenQASM 2.0 source and builds a CUDA-Q kernel."""

    def __init__(self):
        self.kernel = None
        self.qregs = {}  # name -> list of QuakeValues
        self.cregs = {}  # name -> size (declared but unused at runtime)
        self.gate_defs = {}

    def translate(self, source):
        source = _strip_comments(source)
        source = self._consume_header(source)
        source, self.gate_defs = _extract_gate_defs(source)

        self.kernel = make_kernel()

        for raw in _split_top_level(source, ';'):
            stmt = raw.strip()
            if stmt:
                self._execute(stmt)
        return self.kernel

    def _consume_header(self, source):
        m = re.search(r'\bOPENQASM\s+([\d.]+)\s*;', source)
        if not m:
            raise ValueError(
                "Missing 'OPENQASM <version>;' header (required for QASM 2.0).")
        version = m.group(1)
        if not version.startswith('2'):
            raise NotImplementedError(
                f"OpenQASM version {version} is not handled by the 2.x "
                f"translator. Use `from_qasm_str` which dispatches on version.")
        # Drop the header and any includes (we ship built-in gate handlers).
        source = source[:m.start()] + source[m.end():]
        source = re.sub(r'\binclude\s+"[^"]*"\s*;', '', source)
        return source

    def _execute(self, stmt):
        # Register declarations
        m = re.match(r'qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$', stmt)
        if m:
            size = int(m.group(2))
            reg = self.kernel.qalloc(size)
            # Materialise individual qubits eagerly: `QuakeValue` is not
            # iterable in a bounded way, so indexing into it explicitly is
            # what lets `_resolve_qubits` return a concrete Python list.
            self.qregs[m.group(1)] = [reg[i] for i in range(size)]
            return
        m = re.match(r'creg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$', stmt)
        if m:
            self.cregs[m.group(1)] = int(m.group(2))
            return

        # Flow-only / no-op statements
        if re.match(r'barrier\b', stmt):
            return

        # `opaque NAME[(params)] args;` — forward declaration only, the
        # backend supplies semantics; nothing to do at translation time.
        if re.match(r'opaque\b', stmt):
            return

        # measure Q -> C
        m = re.match(r'measure\s+(.+?)\s*->\s*(.+)$', stmt)
        if m:
            for q_ref in _split_top_level(m.group(1), ','):
                for q in self._resolve_qubits(q_ref):
                    self.kernel.mz(q)
            return

        # reset Q
        m = re.match(r'reset\s+(.+)$', stmt)
        if m:
            for q_ref in _split_top_level(m.group(1), ','):
                for q in self._resolve_qubits(q_ref):
                    self.kernel.reset(q)
            return

        # if (c == N) OP
        if re.match(r'if\s*\(', stmt):
            raise NotImplementedError(
                "Classical 'if' branching is not supported by the QASM 2.0 "
                "native parser.")

        # Generic op: NAME[(PARAMS)] Q0, Q1, ...
        name, param_exprs, q_refs = _parse_op_statement(stmt)
        params = [_eval_expr(p) for p in param_exprs]
        qubit_lists = [self._resolve_qubits(q) for q in q_refs]
        self._apply_broadcasted(name, qubit_lists, params)

    def _resolve_qubits(self, ref):
        """'q' → whole register; 'q[3]' → [qubit3]."""
        ref = ref.strip()
        m = re.match(r'([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*$', ref)
        if m:
            name, idx = m.group(1), int(m.group(2))
            if name not in self.qregs:
                raise ValueError(f"Unknown qreg: {name!r}")
            return [self.qregs[name][idx]]
        m = re.match(r'([A-Za-z_]\w*)\s*$', ref)
        if m:
            name = m.group(1)
            if name not in self.qregs:
                raise ValueError(f"Unknown qreg: {name!r}")
            # `self.qregs[name]` is already a list of individual qubits.
            return list(self.qregs[name])
        raise ValueError(f"Cannot parse qubit reference: {ref!r}")

    def _apply_broadcasted(self, name, qubit_lists, params):
        """QASM 2.0 register broadcasting.

        `h q;` → apply h to every qubit in q. If some arguments are single
        qubits and others are registers, broadcast the single qubits across
        all iterations; registers of differing lengths are rejected.
        """
        if not qubit_lists:
            raise ValueError(f"Gate '{name}' has no qubit arguments")
        multi = [len(ql) for ql in qubit_lists if len(ql) > 1]
        if multi and any(l != multi[0] for l in multi):
            raise ValueError(
                f"Gate '{name}' called with mismatched register sizes: {multi}")
        iters = multi[0] if multi else 1
        for i in range(iters):
            qs = [ql[i] if len(ql) > 1 else ql[0] for ql in qubit_lists]
            self._apply_one(name, qs, params)

    def _apply_one(self, name, qs, params):
        """Apply a single gate instance (no broadcasting)."""
        handler = _GATE_HANDLERS.get(name)
        if handler is not None:
            handler(self.kernel, qs, params)
            return
        if name == 'U':
            if len(params) != 3:
                raise ValueError(
                    f"'U' expects 3 parameters in QASM 2.0, got {len(params)}")
            _GATE_HANDLERS['u3'](self.kernel, qs, params)
            return
        if name == 'CX':
            _GATE_HANDLERS['cx'](self.kernel, qs, params)
            return
        # u0(gamma) is idle / duration-only → no-op
        if name == 'u0':
            return
        if name in self.gate_defs:
            self._call_custom_gate(name, qs, params)
            return
        raise ValueError(f"Unsupported gate: {name!r}")

    def _call_custom_gate(self, name, qs, params):
        defn = self.gate_defs[name]
        if len(qs) != len(defn['args']):
            raise ValueError(
                f"Gate '{name}' expects {len(defn['args'])} qubit args, "
                f"got {len(qs)}")
        if len(params) != len(defn['params']):
            raise ValueError(
                f"Gate '{name}' expects {len(defn['params'])} params, "
                f"got {len(params)}")
        qubit_env = dict(zip(defn['args'], qs))
        param_env = dict(zip(defn['params'], params))
        for sub_name, sub_param_exprs, sub_arg_names in defn['body']:
            sub_params = [_eval_expr(e, param_env) for e in sub_param_exprs]
            sub_qs = []
            for a in sub_arg_names:
                a = a.strip()
                if a not in qubit_env:
                    raise ValueError(
                        f"Gate '{name}' body references undefined qubit {a!r}")
                sub_qs.append(qubit_env[a])
            self._apply_one(sub_name, sub_qs, sub_params)


class _QASM3Translator(_QASM2Translator):
    """Parses OpenQASM 3.0 source and builds a CUDA-Q kernel."""

    _MODIFIER_RE = re.compile(r'^(?:ctrl|negctrl|inv|pow)(?:\s*\([^)]*\))?\s*@')

    _CLASSICAL_FEATURES_RE = re.compile(
        r'^(?:def|for|while|let|const|input|output|extern|return|'
        r'break|continue|switch|box|int|uint|float|bool|complex|'
        r'angle|duration|stretch|array)\b')

    def _consume_header(self, source):
        m = re.search(r'\bOPENQASM\s+([\d.]+)\s*;', source)
        if not m:
            raise ValueError(
                "Missing 'OPENQASM <version>;' header (required for QASM 3.0).")
        version = m.group(1)
        if not version.startswith('3'):
            raise NotImplementedError(
                f"OpenQASM version {version} is not handled by the 3.x "
                f"translator. Use `from_qasm_str` which dispatches on version.")
        source = source[:m.start()] + source[m.end():]
        source = re.sub(r'\binclude\s+"[^"]*"\s*;', '', source)
        return source

    def _execute(self, stmt):
        # `qubit[N] name;`
        m = re.match(r'qubit\s*\[\s*(\d+)\s*\]\s+([A-Za-z_]\w*)\s*$', stmt)
        if m:
            size = int(m.group(1))
            reg = self.kernel.qalloc(size)
            self.qregs[m.group(2)] = [reg[i] for i in range(size)]
            return
        # `qubit name;` (single qubit)
        m = re.match(r'qubit\s+([A-Za-z_]\w*)\s*$', stmt)
        if m:
            reg = self.kernel.qalloc(1)
            self.qregs[m.group(1)] = [reg[0]]
            return

        # `bit[N] name;` / `bit name;` — classical registers, tracked but unused.
        m = re.match(r'bit\s*\[\s*(\d+)\s*\]\s+([A-Za-z_]\w*)\s*$', stmt)
        if m:
            self.cregs[m.group(2)] = int(m.group(1))
            return
        m = re.match(r'bit\s+([A-Za-z_]\w*)\s*$', stmt)
        if m:
            self.cregs[m.group(1)] = 1
            return

        # `c = measure q;` or `c[i] = measure q[i];` — classical LHS ignored.
        m = re.match(r'[A-Za-z_]\w*(?:\s*\[\s*\d+\s*\])?\s*=\s*measure\s+(.+)$',
                     stmt)
        if m:
            for q_ref in _split_top_level(m.group(1), ','):
                for q in self._resolve_qubits(q_ref):
                    self.kernel.mz(q)
            return

        # `gphase(γ);` — built-in global phase, no qubits → no-op for sampling.
        if re.match(r'gphase\s*\(', stmt):
            return

        # Reject gate modifiers (`ctrl @`, `ctrl(n) @`, `inv @`, `pow(n) @`).
        if self._MODIFIER_RE.match(stmt):
            raise NotImplementedError(
                f"Gate modifiers (ctrl/negctrl/inv/pow @) are not supported "
                f"by the native parser: {stmt!r}")

        # Reject classical control flow / typed-variable declarations.
        if self._CLASSICAL_FEATURES_RE.match(stmt):
            raise NotImplementedError(
                f"OpenQASM 3.0 classical feature is not supported by the "
                f"native parser: {stmt!r}")

        super()._execute(stmt)

    def _apply_one(self, name, qs, params):
        # Built-in `U(θ, φ, λ)` — or `U(θ, φ, λ, γ)` where γ is a global phase.
        if name == 'U':
            if len(params) not in (3, 4):
                raise ValueError(
                    f"'U' expects 3 or 4 parameters, got {len(params)}")
            return super()._apply_one('u3', qs, params[:3])
        # Legacy uppercase CX (3.0 `stdgates.inc` uses lowercase `cx`).
        if name == 'CX':
            return super()._apply_one('cx', qs, params)
        # Uppercase `I` identity (the 3.0 tables commonly spell it this way;
        # `stdgates.inc` as used by Qiskit exports `id` lowercase, so support both).
        if name == 'I':
            return super()._apply_one('id', qs, params)
        return super()._apply_one(name, qs, params)


def from_qasm_str(qasm_source):
    """Create a CUDA-Q kernel from an OpenQASM 2.0 or 3.0 source string.

    The dispatch is based on the `OPENQASM <version>;` header: `2.x` is
    handled by the QASM 2.0 translator, `3.x` by the QASM 3.0 translator.

    Args:
        `qasm_source`: OpenQASM source (2.x or 3.x) as a string.

    Returns:
        A CUDA-Q kernel equivalent to the QASM circuit.

    Raises:
        ValueError: If the QASM source is malformed, references an
            unsupported gate, or is missing the `OPENQASM` header.
        NotImplementedError: If the QASM version is neither 2.x nor 3.x, or
            the source uses unsupported features (classical `if` branching,
            gate modifiers, typed-variable declarations, subroutines, etc.).

    Supported OpenQASM 2.0 gates (`qelib1.inc` standard library):
        - Paulis / 1-qubit Clifford: ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``,
          ``t``, ``tdg``, ``id``
        - Universal / phase: ``u1``, ``u2``, ``u3``, ``u`` (alias), ``p``,
          ``u0`` (treated as no-op — idle with duration)
        - Rotations: ``rx``, ``ry``, ``rz``
        - Extended 1-qubit: ``sx``, ``sxdg``, ``r`` (Bloch rotation)
        - Controlled 1-qubit: ``cx``, ``cy``, ``cz``, ``ch``, ``cs``, ``csdg``,
          ``csx``, ``crx``, ``cry``, ``crz``, ``cp``, ``cu1``, ``cu3``, ``cu``
        - Swaps / exchange: ``swap``, ``iswap``, ``dcx``, ``ecr``, ``cswap``
        - Parametric two-qubit: ``rxx``, ``ryy``, ``rzz``, ``rzx``
        - Multi-qubit: ``ccx`` (Toffoli), ``ccz``, ``rccx``, ``c3x``, ``c4x``,
          ``mcx``, ``mcp``/``mcphase``
        - Non-gate statements: ``measure``, ``reset``, ``barrier``
        - User-defined ``gate NAME(params) args { body }`` with recursive
          expansion (including nested custom gates).

    Supported OpenQASM 3.0 features (on top of the 2.0 gate set):
        - Declarations: ``qubit[N] q;``, ``qubit q;``, ``bit[N] c;``, ``bit c;``
        - Measurement assignment: ``c = measure q;`` / ``c[i] = measure q[i];``
        - Built-in gates: ``U(θ,φ,λ)`` (and 4-parameter form ``U(θ,φ,λ,γ)``),
          ``gphase(γ)`` (no-op for sampling), ``CX`` (legacy uppercase)
        - ``include "stdgates.inc";`` (handled by our gate table)

    Not supported:
        - Classical control flow (``if (c==N) OP;``)
        - Gate modifiers: ``ctrl @``, ``negctrl @``, ``inv @``, ``pow(n) @``
        - Classical types and control: ``def``, ``for``, ``while``, ``input``,
          ``output``, ``let``, ``const``, ``int``/``float``/``angle``/etc.
    """
    stripped = _strip_comments(qasm_source)
    m = re.search(r'\bOPENQASM\s+([\d.]+)\s*;', stripped)
    if not m:
        raise ValueError("Missing 'OPENQASM <version>;' header.")
    version = m.group(1)
    if version.startswith('2'):
        return _QASM2Translator().translate(qasm_source)
    if version.startswith('3'):
        return _QASM3Translator().translate(qasm_source)
    raise NotImplementedError(
        f"OpenQASM version {version} is not supported (only 2.x and 3.x).")


def from_qasm(qasm_file):
    """Create a CUDA-Q kernel from an OpenQASM 2.0 or 3.0 file.

    Args:
        `qasm_file`: Path to the OpenQASM file as a string.

    Returns:
        A CUDA-Q kernel equivalent to the OpenQASM circuit.

    Raises:
        FileNotFoundError: If the QASM file does not exist.
        ValueError: If the QASM file cannot be parsed or references an
            unsupported gate.
        NotImplementedError: If the QASM version is neither 2.x nor 3.x, or
            the source uses unsupported features.
    """
    with open(qasm_file, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        return from_qasm_str(source)
    except (ValueError, NotImplementedError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Could not parse QASM file '{qasm_file}': {e}") from e
