# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from ..algebra.gf2 import GF2Matrix


@dataclass(frozen=True, slots=True)
class GF2Partition:
    """One named direct-sum component of a binary linear map."""

    name: str
    width: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("GF(2) partition names must be nonempty strings")
        if (not isinstance(self.width, int) or isinstance(self.width, bool) or
                self.width < 0):
            raise TypeError("GF(2) partition widths must be nonnegative ints")


@dataclass(frozen=True, slots=True)
class PartitionedGF2Map:
    """Affine GF(2) map with named row and column direct-sum partitions.

    This is the small Python algebra view used by code profiles, gadget
    code algebra, gadget predicates, and symbolic propagation.  It is not a
    second execution IR: compiler passes derive it from canonical Fabric IR
    and may lower it to the block/symbolic contraction implementation.
    """

    matrix: GF2Matrix
    rows: tuple[GF2Partition, ...]
    columns: tuple[GF2Partition, ...]
    shift: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, GF2Matrix):
            raise TypeError(
                "PartitionedGF2Map.matrix must be a cudaq.logical.GF2Matrix")
        rows = tuple(self.rows)
        columns = tuple(self.columns)
        if any(not isinstance(item, GF2Partition)
               for item in (*rows, *columns)):
            raise TypeError(
                "map partitions must be cudaq.logical.GF2Partition values")
        if len({item.name for item in rows}) != len(rows):
            raise ValueError("row partition names must be unique")
        if len({item.name for item in columns}) != len(columns):
            raise ValueError("column partition names must be unique")
        if sum(item.width for item in rows) != self.matrix.nrows:
            raise ValueError("row partitions do not cover the matrix rows")
        if sum(item.width for item in columns) != self.matrix.ncols:
            raise ValueError(
                "column partitions do not cover the matrix columns")
        shift = tuple(int(value) for value in self.shift)
        if not shift:
            shift = (0,) * self.matrix.nrows
        if len(shift) != self.matrix.nrows or any(
                value not in (0, 1) for value in shift):
            raise ValueError(
                "affine shift must contain one binary value per row")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "shift", shift)

    @staticmethod
    def _range(partitions, name: str) -> range:
        offset = 0
        for partition in partitions:
            if partition.name == name:
                return range(offset, offset + partition.width)
            offset += partition.width
        raise KeyError(name)

    def block(self, row: str, column: str) -> GF2Matrix:
        """Return one dense block selected by its semantic partition names."""

        row_range = self._range(self.rows, row)
        column_range = self._range(self.columns, column)
        return GF2Matrix(
            tuple(
                tuple(self.matrix.rows[i][j]
                      for j in column_range)
                for i in row_range),
            ncols=len(column_range),
        )

    def row_shift(self, row: str) -> tuple[int, ...]:
        selected = self._range(self.rows, row)
        return tuple(self.shift[index] for index in selected)

    def symbolic(self, interner=None) -> "SymbolicPartitionedGF2Map":
        """Intern this concrete map as a reusable symbolic block grid."""

        return SymbolicPartitionedGF2Map.from_concrete(self, interner=interner)


@dataclass(frozen=True, slots=True)
class GF2BlockVariable:
    """One interned concrete block referenced by symbolic expressions."""

    index: int
    value: GF2Matrix

    @property
    def nrows(self):
        return self.value.nrows

    @property
    def ncols(self):
        return self.value.ncols


class GF2BlockInterner:
    """Deduplicate equal dense blocks across gadget calls and sweeps."""

    def __init__(self):
        self._lock = RLock()
        self._by_value = {}
        self._variables = []

    def intern(self, value: GF2Matrix) -> GF2BlockVariable:
        if not isinstance(value, GF2Matrix):
            raise TypeError("only GF2Matrix blocks can be interned")
        key = (value.nrows, value.ncols, value.rows)
        with self._lock:
            variable = self._by_value.get(key)
            if variable is None:
                variable = GF2BlockVariable(len(self._variables), value)
                self._variables.append(variable)
                self._by_value[key] = variable
            return variable

    @property
    def variables(self):
        return tuple(self._variables)


@dataclass(frozen=True, slots=True)
class GF2BlockMonomial:
    """Ordered non-commutative product of interned GF(2) blocks."""

    factors: tuple[GF2BlockVariable, ...]

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        if not factors:
            raise ValueError("a block monomial requires at least one factor")
        for left, right in zip(factors, factors[1:]):
            if left.ncols != right.nrows:
                raise ValueError(
                    "block-monomial factor dimensions do not compose")
        object.__setattr__(self, "factors", factors)

    @property
    def nrows(self):
        return self.factors[0].nrows

    @property
    def ncols(self):
        return self.factors[-1].ncols

    def evaluate(self) -> GF2Matrix:
        value = self.factors[0].value
        for factor in self.factors[1:]:
            value = value @ factor.value
        return value


@dataclass(frozen=True, slots=True)
class GF2BlockPolynomial:
    """GF(2) sum of ordered block monomials with duplicate cancellation."""

    nrows: int
    ncols: int
    terms: frozenset[GF2BlockMonomial] = frozenset()

    def __post_init__(self) -> None:
        if self.nrows < 0 or self.ncols < 0:
            raise ValueError("block-polynomial dimensions must be nonnegative")
        terms = frozenset(self.terms)
        if any(term.nrows != self.nrows or term.ncols != self.ncols
               for term in terms):
            raise ValueError("block-polynomial term dimensions disagree")
        object.__setattr__(self, "terms", terms)

    @classmethod
    def variable(cls, variable):
        monomial = GF2BlockMonomial((variable,))
        return cls(variable.nrows, variable.ncols, frozenset((monomial,)))

    def __xor__(self, other):
        if not isinstance(other, GF2BlockPolynomial):
            return NotImplemented
        if (self.nrows, self.ncols) != (other.nrows, other.ncols):
            raise ValueError("only equal-shaped block polynomials can be XORed")
        return GF2BlockPolynomial(self.nrows, self.ncols,
                                  self.terms.symmetric_difference(other.terms))

    def compose(self, other):
        """Return ``self @ other`` without evaluating dense products."""

        if not isinstance(other, GF2BlockPolynomial):
            raise TypeError(
                "block-polynomial composition requires another polynomial")
        if self.ncols != other.nrows:
            raise ValueError("block-polynomial dimensions do not compose")
        terms = set()
        for left in self.terms:
            for right in other.terms:
                term = GF2BlockMonomial((*left.factors, *right.factors))
                terms.remove(term) if term in terms else terms.add(term)
        return GF2BlockPolynomial(self.nrows, other.ncols, frozenset(terms))

    def evaluate(self) -> GF2Matrix:
        rows = [[0] * self.ncols for _ in range(self.nrows)]
        for term in self.terms:
            value = term.evaluate()
            for row in range(self.nrows):
                for column in range(self.ncols):
                    rows[row][column] ^= value.rows[row][column]
        return GF2Matrix(tuple(tuple(row) for row in rows), ncols=self.ncols)


@dataclass(frozen=True, slots=True)
class SymbolicPartitionedGF2Map:
    """Named block grid whose cells are deferred GF(2) polynomials."""

    rows: tuple[GF2Partition, ...]
    columns: tuple[GF2Partition, ...]
    blocks: tuple[tuple[GF2BlockPolynomial, ...], ...]
    interner: GF2BlockInterner

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        columns = tuple(self.columns)
        blocks = tuple(tuple(row) for row in self.blocks)
        if len(blocks) != len(rows) or any(
                len(row) != len(columns) for row in blocks):
            raise ValueError(
                "symbolic block grid shape disagrees with partitions")
        for i, row in enumerate(blocks):
            for j, block in enumerate(row):
                if (block.nrows, block.ncols) != (rows[i].width,
                                                  columns[j].width):
                    raise ValueError(
                        "symbolic block dimensions disagree with their slot")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "blocks", blocks)

    @classmethod
    def from_concrete(cls, value, *, interner=None):
        if not isinstance(value, PartitionedGF2Map):
            raise TypeError("symbolic conversion requires PartitionedGF2Map")
        if any(value.shift):
            raise NotImplementedError(
                "symbolic affine shifts require homogeneous lifting")
        interner = interner or GF2BlockInterner()
        blocks = []
        for row in value.rows:
            block_row = []
            for column in value.columns:
                concrete = value.block(row.name, column.name)
                if any(any(item for item in items) for items in concrete.rows):
                    block_row.append(
                        GF2BlockPolynomial.variable(interner.intern(concrete)))
                else:
                    block_row.append(GF2BlockPolynomial(row.width,
                                                        column.width))
            blocks.append(tuple(block_row))
        return cls(value.rows, value.columns, tuple(blocks), interner)

    def compose(self, other):
        """Symbolically compose two fully matched direct-sum interfaces."""

        if not isinstance(other, SymbolicPartitionedGF2Map):
            raise TypeError(
                "symbolic composition requires another symbolic map")
        if self.interner is not other.interner:
            raise ValueError("symbolic maps must share one block interner")
        if len(self.columns) != len(other.rows) or any(
                left.width != right.width or left.name != right.name
                for left, right in zip(self.columns, other.rows)):
            raise ValueError("symbolic direct-sum interfaces do not match")
        blocks = []
        for i, row in enumerate(self.rows):
            block_row = []
            for k, column in enumerate(other.columns):
                value = GF2BlockPolynomial(row.width, column.width)
                for j in range(len(self.columns)):
                    value = value ^ self.blocks[i][j].compose(
                        other.blocks[j][k])
                block_row.append(value)
            blocks.append(tuple(block_row))
        return SymbolicPartitionedGF2Map(self.rows, other.columns,
                                         tuple(blocks), self.interner)

    def evaluate(self) -> PartitionedGF2Map:
        row_offsets = []
        offset = 0
        for part in self.rows:
            row_offsets.append(offset)
            offset += part.width
        column_offsets = []
        offset = 0
        for part in self.columns:
            column_offsets.append(offset)
            offset += part.width
        dense = [
            [0] * offset for _ in range(sum(part.width for part in self.rows))
        ]
        for i, block_row in enumerate(self.blocks):
            for j, polynomial in enumerate(block_row):
                value = polynomial.evaluate()
                for local_row in range(value.nrows):
                    for local_column in range(value.ncols):
                        dense[row_offsets[i] + local_row][
                            column_offsets[j] +
                            local_column] = value.rows[local_row][local_column]
        return PartitionedGF2Map(
            GF2Matrix(tuple(tuple(row) for row in dense), ncols=offset),
            self.rows,
            self.columns,
        )


def compose_symbolic_chain(
    maps,
    *,
    strategy="forward",
    interner=None,
    window_size=8,
):
    """Compose a chain with forward, balanced, or bounded-window evaluation.

    ``windowed`` periodically evaluates and re-interns the live boundary map,
    bounding symbolic monomial depth while preserving block reuse. Symbolic
    graph pruning can additionally harvest finished observation wires; this
    helper is the lower-level fully matched seam primitive.
    """

    values = tuple(maps)
    if not values:
        raise ValueError("symbolic composition requires a nonempty chain")
    interner = interner or GF2BlockInterner()

    def symbolic(value):
        if isinstance(value, PartitionedGF2Map):
            return value.symbolic(interner)
        if isinstance(value, SymbolicPartitionedGF2Map):
            if value.interner is not interner:
                raise ValueError(
                    "all symbolic maps must share the requested interner")
            return value
        raise TypeError(
            "chain entries must be concrete or symbolic partitioned maps")

    values = tuple(symbolic(value) for value in values)
    if strategy == "forward":
        result = values[0]
        for value in values[1:]:
            result = value.compose(result)
        return result
    if strategy == "balanced":
        level = list(values)
        while len(level) > 1:
            next_level = []
            for index in range(0, len(level), 2):
                if index + 1 == len(level):
                    next_level.append(level[index])
                else:
                    next_level.append(level[index + 1].compose(level[index]))
            level = next_level
        return level[0]
    if strategy == "windowed":
        if (not isinstance(window_size, int) or isinstance(window_size, bool) or
                window_size <= 0):
            raise ValueError("window_size must be a positive int")
        result = values[0]
        depth = 1
        for value in values[1:]:
            result = value.compose(result)
            depth += 1
            if depth == window_size:
                result = result.evaluate().symbolic(interner)
                depth = 1
        return result
    raise ValueError("strategy must be forward, balanced, or windowed")


__all__ = [
    "GF2BlockInterner",
    "GF2BlockMonomial",
    "GF2BlockPolynomial",
    "GF2BlockVariable",
    "GF2Partition",
    "PartitionedGF2Map",
    "SymbolicPartitionedGF2Map",
    "compose_symbolic_chain",
]
