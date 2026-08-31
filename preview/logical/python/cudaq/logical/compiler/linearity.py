# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Linear-use verification over canonical CUDA-Q Logical / LVM / Fabric bodies.

The CUDA-Q Logical ownership contract makes logical qubits, encoded patches, consumable
resource states, linear events, and
value-semantic frames *linear*: ordinary MLIR SSA permits arbitrarily many
uses, so a dedicated analysis must verify that every linear SSA value has
exactly one owner along every execution path.  This module implements that
analysis in Python over the canonical IR using the
``QLXLinearOpInterface`` classification:

- ``consume``   -- the default for any use of a linear operand.  Transforming
  operations (gates, measurement reads, calls, terminators, region carries)
  take ownership; producing a successor value is the op's business, not the
  checker's.
- ``borrow``    -- observation-only uses that never take ownership.  These are
  the event observers (``event_test`` / ``event_poll`` /
  ``event_select_ready``) and Pauli-frame resolution
  (``fabric.frame_resolve``), whose op definitions state they do not consume
  their operand.

Region awareness:

- ``if``-like operations (``qlx.if`` / ``lvm.if`` / ``fabric.if`` and the
  three-way ``event_try_take``) execute exactly one of
  their regions, so one consuming use per branch region is a single dynamic
  consume (the canonical Pauli-byproduct pattern consumes the same qubit in
  both the then- and else-branch).
- Loop-like operations (``repeat`` / ``while``) execute their regions many
  times; consuming a value captured from outside the loop is therefore always
  a violation -- linear state must enter a loop as an explicit carry.
- Any other region-bearing op is treated as a transparent run-once scope.

The checker is deliberately conservative in what it *reports* (no false
positives on legal single-owner IR) while catching the two real bug classes:
double consumption (including use-after-destructive-measure) and consumption
inside a loop that does not own the value.  A linear value with no consuming
use at all is reported as a leak for types that have explicit disposal ops;
frame types, which have no dispose op and end their life by being dropped,
are exempt.
"""

from __future__ import annotations

from dataclasses import dataclass

# Symbol-level operations whose bodies form independent linear-ownership
# scopes.  Each is verified on its own; nested occurrences (e.g. a factory
# gadget declared inside a fabric.machine region) are not double-walked.
BODY_OPS = frozenset({
    "qlx.program",
    "qlx.objective_body",
    "lvm.kernel",
    "fabric.gadget",
    "fabric.protocol",
    "fabric.circuit",
})

# Exactly one region executes: consuming uses in *different* regions of the
# same op describe exclusive paths and together form one dynamic consume.
BRANCH_EXCLUSIVE_OPS = frozenset({
    "qlx.if",
    "lvm.if",
    "fabric.if",
    "qlx.event_try_take",
    "lvm.event_try_take",
    "fabric.event_try_take",
})

# Regions execute repeatedly: linear state may only enter as an explicit
# initial or loop-carried value (which the operation consumes); consuming a
# captured outer value inside is a per-iteration double consume.
LOOP_OPS = frozenset({
    "qlx.repeat",
    "qlx.while",
    "lvm.repeat",
    "lvm.while",
    "fabric.repeat",
    "fabric.while",
})

# Observation-only operations.  Their op definitions promise they do not take
# payload ownership ("observe ... without consuming", "select one ready event
# without consuming payload ownership", "resolve measurement through frame").
BORROW_OPS = frozenset({
    f"{dialect}.{name}" for dialect in ("qlx", "lvm", "fabric")
    for name in ("event_test", "event_poll", "event_select_ready")
}) | frozenset({"fabric.frame_resolve"})

# Linear types identified by their dialect type head (the text before any
# `<...>` parameter list).  These are linear unconditionally.
LINEAR_TYPE_HEADS = frozenset({
    "!qlx.logical_qubit",
    "!qlx.lqbit",
    "!qlx.logical_resource",
    "!lvm.logical_qubit",
    "!lvm.logical_resource",
    "!fabric.patch",
    "!fabric.patch_frame",
    "!fabric.patch_bundle",
    "!fabric.resource",
})

# Event types carry an explicit ownership parameter; only "linear" events are
# single-owner values.
EVENT_TYPE_HEADS = frozenset({
    "!qlx.logical_event",
    "!lvm.logical_event",
    "!fabric.event",
})

# Value-semantic frame state is threaded linearly (each update consumes the
# previous frame) but has no disposal op: the terminal frame is legitimately
# dropped, so frames are exempt from the leak check.
DROPPABLE_TYPE_HEADS = frozenset({
    "!qlx.logical_frame",
    "!lvm.logical_frame",
    "!fabric.frame",
})


def _type_head(type_text: str) -> str:
    return type_text.split("<", 1)[0]


def is_linear_type(mlir_type) -> bool:
    """True when values of this type are single-owner linear values."""
    text = str(mlir_type)
    head = _type_head(text)
    if head in LINEAR_TYPE_HEADS or head in DROPPABLE_TYPE_HEADS:
        return True
    if head in EVENT_TYPE_HEADS:
        # The ownership string is the trailing type parameter.
        return text.replace(" ", "").endswith('"linear">')
    return False


def _may_drop(mlir_type) -> bool:
    return _type_head(str(mlir_type)) in DROPPABLE_TYPE_HEADS


@dataclass(frozen=True, slots=True)
class LinearityViolation:
    """One linear-ownership violation at a concrete IR location."""

    body: str
    value: str
    kind: str  # double-consume | loop-capture-consume | borrow-after-consume | unconsumed
    message: str
    location: str

    def render(self) -> str:
        return f"[{self.kind}] {self.body}: {self.message} at {self.location}"


@dataclass(frozen=True, slots=True)
class LinearityReport:
    """Structured result of one linear-use verification run."""

    checked_bodies: tuple[str, ...]
    linear_values: int
    violations: tuple[LinearityViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def result(self) -> str:
        if not self.checked_bodies:
            return "skipped"
        return "pass" if self.ok else "fail"


class LinearityError(ValueError):
    """Linear single-ownership verification failed."""

    def __init__(self, report: LinearityReport, subject: str | None = None):
        self.report = report
        header = "linear ownership verification failed"
        if subject:
            header += f" for {subject}"
        lines = [header] + [
            "  - " + violation.render() for violation in report.violations
        ]
        super().__init__("\n".join(lines))


def _symbol_of(operation) -> str:
    try:
        attr = operation.attributes["sym_name"]
    except KeyError:
        return operation.name
    return f"{operation.name} @" + str(getattr(attr, "value", attr)).strip('"')


@dataclass(frozen=True, slots=True)
class _Use:
    owner: object  # Operation
    operand_number: int
    # (ancestor Operation, region index) steps from the value's defining
    # scope down to (but excluding) the owning op.
    chain: tuple


def _describe_value(value, index_hint: str) -> str:
    try:
        name = value.get_name()
    except Exception:  # pragma: no cover - defensive
        name = "<value>"
    return f"{name} : {value.type} ({index_hint})"


class _BodyAnalysis:
    """Linear-use analysis of one single-ownership body scope."""

    def __init__(self, body_operation):
        self.body = body_operation
        self.body_symbol = _symbol_of(body_operation)
        # Operation -> tuple of (ancestor Operation, region index) from the
        # body root down to (excluding) the operation itself.
        self._paths: dict[object, tuple] = {}
        self._names = {body_operation: body_operation.name}
        self._definitions: list[tuple] = []
        self.violations: list[LinearityViolation] = []
        self.linear_values = 0
        self._collect()

    # -- IR walk -----------------------------------------------------------

    def _collect(self) -> None:
        """Index operation paths and linear definitions in one IR traversal."""

        stack = [(self.body, ())]
        while stack:
            operation, path = stack.pop()
            operation_name = self._names[operation]
            for region_index, region in enumerate(operation.regions):
                scope = path + ((operation, region_index),)
                for block in region.blocks:
                    for argument in block.arguments:
                        if is_linear_type(argument.type):
                            self._definitions.append((
                                argument,
                                scope,
                                f"block argument {argument.arg_number} of "
                                f"{operation_name}",
                            ))
                    for child in block.operations:
                        child_operation = child.operation
                        child_name = child_operation.name
                        self._names[child_operation] = child_name
                        self._paths[child_operation] = scope
                        if child_name in BODY_OPS:
                            continue
                        for result in child_operation.results:
                            if is_linear_type(result.type):
                                self._definitions.append((
                                    result,
                                    scope,
                                    f"result of {child_name}",
                                ))
                        stack.append((child_operation, scope))

    def _name(self, operation) -> str:
        name = self._names.get(operation)
        if name is None:
            name = operation.name
            self._names[operation] = name
        return name

    # -- ownership analysis ------------------------------------------------

    def run(self) -> None:
        for value, scope, index_hint in self._definitions:
            self.linear_values += 1
            self._check_value(value, scope, index_hint)

    def _relative_uses(self, value, scope):
        borrows, consumes = [], []
        for use in value.uses:
            owner = use.owner
            owner_operation = getattr(owner, "operation", owner)
            full_chain = self._paths.get(owner_operation)
            if full_chain is None:
                # Use outside this body's walk (cannot happen for verified
                # SSA); treat as consuming at top level to stay conservative.
                full_chain = scope
            chain = full_chain[len(scope):]
            record = _Use(owner_operation, use.operand_number, chain)
            if self._name(owner_operation) in BORROW_OPS:
                borrows.append(record)
            else:
                consumes.append(record)
        return borrows, consumes

    def _check_value(self, value, scope, index_hint) -> None:
        borrows, consumes = self._relative_uses(value, scope)
        description = None

        def describe():
            nonlocal description
            if description is None:
                description = _describe_value(value, index_hint)
            return description

        # Rule 1: no consuming use may sit inside a loop region that does not
        # own the value -- it would re-consume the same owner every iteration.
        for use in consumes:
            loop = next(
                (step for step in use.chain if self._name(step[0]) in LOOP_OPS),
                None,
            )
            if loop is not None:
                self._report(
                    "loop-capture-consume",
                    f"linear value {describe()} defined outside "
                    f"{self._name(loop[0])} is consumed inside its body by "
                    f"{self._name(use.owner)}; loop-carried linear state must be "
                    "threaded as an explicit init/carry",
                    use.owner,
                )

        # Rule 2: at most one dynamic consume on every path.
        for index, first in enumerate(consumes):
            for second in consumes[index + 1:]:
                if self._conflicts(first, second):
                    first, second = self._program_order(first, second)
                    self._report(
                        "double-consume",
                        f"linear value {describe()} is consumed more than "
                        f"once: by {self._name(first.owner)} "
                        f"(at {first.owner.location}) and again by "
                        f"{self._name(second.owner)}",
                        second.owner,
                    )

        # Rule 3: observation must precede consumption on the same path.
        for borrow in borrows:
            for consume in consumes:
                if self._borrow_after_consume(borrow, consume):
                    self._report(
                        "borrow-after-consume",
                        f"linear value {describe()} is observed by "
                        f"{self._name(borrow.owner)} after being consumed by "
                        f"{self._name(consume.owner)} "
                        f"(at {consume.owner.location})",
                        borrow.owner,
                    )

        # Rule 4: linear values must eventually be consumed.  Frame-typed
        # values have no disposal op and may be dropped.
        if not consumes and not _may_drop(value.type):
            self._report(
                "unconsumed",
                f"linear value {describe()} is never consumed; every "
                "linear owner needs exactly one consuming use "
                "(return/yield it, or dispose of it explicitly)",
                self.body,
            )

    @staticmethod
    def _program_order(first: _Use, second: _Use) -> tuple:
        """Best-effort textual ordering of two conflicting uses."""
        try:
            if (first.owner.block == second.owner.block and
                    second.owner != first.owner and
                    second.owner.is_before_in_block(first.owner)):
                return second, first
        except Exception:  # pragma: no cover - defensive
            pass
        return first, second

    def _conflicts(self, first: _Use, second: _Use) -> bool:
        """True when two consuming uses can both execute for one owner."""
        depth = 0
        while True:
            a = first.chain[depth] if depth < len(first.chain) else None
            b = second.chain[depth] if depth < len(second.chain) else None
            if a is None and b is None:
                # Same region: two sequential consumes (or one op consuming
                # the same value through two operands).
                return True
            if a is None or b is None:
                # One use is the region-bearing ancestor of the other: the
                # outer op consumed the value and an inner op consumes it
                # again while the outer op runs.
                return True
            if a[0] == b[0]:
                if a[1] != b[1]:
                    # Different regions of one op: exclusive only for
                    # if-like control.
                    return self._name(a[0]) not in BRANCH_EXCLUSIVE_OPS
                depth += 1
                continue
            # Divergence at two distinct ops.  In single-block regions they
            # execute sequentially -> conflict. Across control-flow graph
            # blocks, the paths may be exclusive, so stay silent there.
            try:
                same_block = a[0].block == b[0].block
            except Exception:  # pragma: no cover - defensive
                same_block = True
            return same_block

    def _borrow_after_consume(self, borrow: _Use, consume: _Use) -> bool:
        depth = 0
        while True:
            a = borrow.chain[depth] if depth < len(borrow.chain) else None
            b = consume.chain[depth] if depth < len(consume.chain) else None
            top_borrow = a[0] if a is not None else borrow.owner
            top_consume = b[0] if b is not None else consume.owner
            if top_borrow == top_consume:
                if a is None or b is None:
                    # Borrow nested within the consuming op (or vice versa):
                    # simultaneous, not provably after.
                    return False
                if a[1] != b[1]:
                    # Exclusive or sibling regions: not ordered after.
                    return False
                depth += 1
                continue
            try:
                if top_borrow.block != top_consume.block:
                    return False
                return top_consume.is_before_in_block(top_borrow)
            except Exception:  # pragma: no cover - defensive
                return False

    def _report(self, kind: str, message: str, at_operation) -> None:
        self.violations.append(
            LinearityViolation(
                body=self.body_symbol,
                value=message.split(" is ", 1)[0],
                kind=kind,
                message=message,
                location=str(at_operation.location),
            ))


def _is_declaration_only(operation) -> bool:
    """True for vestigial bodies that carry no semantic content.

    A ``fabric.gadget ... realization @circuit`` keeps a declaration-only
    region (an unused boundary signature plus a bare terminator); ownership
    is verified on the referenced ``fabric.circuit`` body instead.
    """
    regions = list(operation.regions)
    if len(regions) != 1:
        return False
    blocks = list(regions[0].blocks)
    if not blocks:
        return True
    if len(blocks) != 1:
        return False
    block = blocks[0]
    operations = list(block.operations)
    if len(operations) > 1:
        return False
    if operations:
        terminator = operations[0].operation
        if (len(terminator.operands) or len(terminator.results) or
                list(terminator.regions)):
            return False
    return all(not list(argument.uses) for argument in block.arguments)


def _iter_body_operations(module):

    def visit(operation):
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    child_operation = child.operation
                    if child_operation.name in BODY_OPS:
                        yield child_operation
                    yield from visit(child_operation)

    yield from visit(module.operation)


def check_linearity(module) -> LinearityReport:
    """Run the linear single-ownership analysis over every body in a module.

    Returns a structured :class:`LinearityReport`; never raises for
    violations.  ``module`` is a ``cudaq.mlir.ir.Module`` (or any operation exposing
    ``.operation`` with regions).
    """
    checked: list[str] = []
    violations: list[LinearityViolation] = []
    linear_values = 0
    for body_operation in _iter_body_operations(module):
        if _is_declaration_only(body_operation):
            continue
        analysis = _BodyAnalysis(body_operation)
        analysis.run()
        checked.append(analysis.body_symbol)
        violations.extend(analysis.violations)
        linear_values += analysis.linear_values
    return LinearityReport(
        checked_bodies=tuple(checked),
        linear_values=linear_values,
        violations=tuple(violations),
    )


def verify_linearity(module, *, subject: str | None = None) -> LinearityReport:
    """Run :func:`check_linearity` and raise :class:`LinearityError` on any
    violation.  Returns the (passing or vacuous) report otherwise."""
    report = check_linearity(module)
    if not report.ok:
        raise LinearityError(report, subject=subject)
    return report
