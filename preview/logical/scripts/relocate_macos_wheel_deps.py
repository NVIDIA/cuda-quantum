#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Make a repaired macOS wheel load its shared libraries from cudaq-core.

``delocate-wheel`` leaves a dependency alone when it matches ``-e``. The
libraries shipped by cudaq-core (GMP and MPFR) are excluded that way, so a
reference to the build machine's copy, such as
``/opt/homebrew/opt/gmp/lib/libgmp.10.dylib``, stays in the wheel and fails
to load on a Mac without that copy.

``rewrite`` turns those references into ``@rpath/<library>``, re-signs the
modified binaries, and regenerates ``RECORD``. ``check`` fails when a binary
depends on anything that is neither ``@``-relative nor a system library.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",  # 64-bit, little endian
    b"\xca\xfe\xba\xbe",  # universal
}
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")


def is_macho(data: bytes) -> bool:
    return data[:4] in MACHO_MAGICS


def dependencies(binary: Path) -> list[str]:
    """Return the load commands of ``binary``, without its own install name."""

    output = subprocess.run(["otool", "-L", str(binary)],
                            check=True,
                            capture_output=True,
                            text=True).stdout
    lines = output.splitlines()
    # A universal static archive lists its members, not load commands.
    if any(line.startswith("Archive :") for line in lines):
        return []
    return [line.split()[0] for line in lines[1:]]


def is_allowed(dependency: str) -> bool:
    return dependency.startswith("@") or dependency.startswith(SYSTEM_PREFIXES)


def unpack_binaries(wheel: Path, directory: Path) -> dict[str, Path]:
    """Extract the Mach-O members of ``wheel`` and map member name to path."""

    binaries = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            data = archive.read(info)
            if not is_macho(data):
                continue
            path = directory / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            binaries[info.filename] = path
    return binaries


def check(wheel: Path) -> bool:
    """Report the binaries of ``wheel`` that load a non-relocatable library."""

    with tempfile.TemporaryDirectory() as scratch:
        binaries = unpack_binaries(wheel, Path(scratch))
        offenders = {
            name: [d for d in dependencies(path) if not is_allowed(d)]
            for name, path in sorted(binaries.items())
        }
    offenders = {name: deps for name, deps in offenders.items() if deps}
    for name, deps in offenders.items():
        print(f"{wheel.name}: {name} loads {', '.join(deps)}", file=sys.stderr)
    print(f"{wheel.name}: checked {len(binaries)} binaries, "
          f"{len(offenders)} with non-relocatable dependencies")
    return not offenders


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def rewrite(wheel: Path, out_dir: Path, libraries: set[str]) -> Path:
    """Point absolute references to ``libraries`` at ``@rpath`` and repack."""

    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / wheel.name
    changed: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory() as scratch:
        for name, path in unpack_binaries(wheel, Path(scratch)).items():
            edits = [
                (dep, f"@rpath/{PurePosixPath(dep).name}")
                for dep in dependencies(path)
                if not is_allowed(dep) and PurePosixPath(dep).name in libraries
            ]
            if not edits:
                continue
            command = ["install_name_tool"]
            for old, new in edits:
                command += ["-change", old, new]
            subprocess.run([*command, str(path)], check=True)
            # install_name_tool invalidates the signature, which arm64 requires.
            subprocess.run(["codesign", "--force", "--sign", "-",
                            str(path)],
                           check=True)
            changed[name] = path.read_bytes()
            print(f"{name}: " + ", ".join(f"{o} -> {n}" for o, n in edits))

    if not changed:
        print(f"{wheel.name}: no reference to {sorted(libraries)} to rewrite")
        shutil.copy2(wheel, result)
        return result

    with zipfile.ZipFile(wheel) as source, \
            zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as target:
        record_name = next(i.filename
                           for i in source.infolist()
                           if i.filename.endswith(".dist-info/RECORD"))
        lines = []
        for info in source.infolist():
            if info.is_dir() or info.filename == record_name:
                continue
            data = changed.get(info.filename, None)
            if data is None:
                data = source.read(info)
            target.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
            lines.append(record_line(info.filename, data))
        lines.append(f"{record_name},,")
        record_info = zipfile.ZipInfo(
            record_name, date_time=source.getinfo(record_name).date_time)
        record_info.external_attr = source.getinfo(record_name).external_attr
        target.writestr(record_info,
                        "\n".join(lines) + "\n",
                        compress_type=zipfile.ZIP_DEFLATED)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    rewrite_parser = commands.add_parser(
        "rewrite", help="rewrite references to libraries shipped in cudaq-core")
    rewrite_parser.add_argument("wheel", type=Path)
    rewrite_parser.add_argument("--out-dir", type=Path, required=True)
    rewrite_parser.add_argument(
        "--library",
        action="append",
        required=True,
        help="file name of a library to load via @rpath")

    check_parser = commands.add_parser(
        "check", help="fail on dependencies that are not @-relative or system")
    check_parser.add_argument("wheels", type=Path, nargs="+")

    args = parser.parse_args()
    if args.command == "rewrite":
        print(rewrite(args.wheel, args.out_dir, set(args.library)))
        return 0
    return 0 if all([check(wheel) for wheel in args.wheels]) else 1


if __name__ == "__main__":
    sys.exit(main())
