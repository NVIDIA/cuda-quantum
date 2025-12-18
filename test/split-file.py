#!/usr/bin/env python3
"""
Minimal split-file implementation for CUDAQ lit tests.

Parses an input test containing `//--- <filename>` markers and emits the
annotated sections into separate files within the designated output directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def split_file(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    current_file = None
    output_handle = None

    with source.open("r", encoding="utf-8") as infile:
        for line in infile:
            if line.startswith("//---"):
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    raise RuntimeError(f"Malformed split marker: {line.strip()}")
                relative_path = parts[1]
                current_file = destination / relative_path
                current_file.parent.mkdir(parents=True, exist_ok=True)
                if output_handle:
                    output_handle.close()
                output_handle = current_file.open("w", encoding="utf-8")
                continue

            if output_handle is None:
                # Ignore content prior to the first split marker.
                continue

            output_handle.write(line)

    if output_handle:
        output_handle.close()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: split-file.py <input> <output-dir>", file=sys.stderr)
        return 1

    input_path = Path(argv[1]).resolve()
    output_dir = Path(argv[2]).resolve()

    if not input_path.exists():
        print(f"error: input file '{input_path}' does not exist", file=sys.stderr)
        return 1

    split_file(input_path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

