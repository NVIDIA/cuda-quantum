#!/usr/bin/env python3

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rewrite a wheel's metadata version without touching compiled binaries.

Usage:
  python scripts/restamp_wheel.py --wheel PATH --version NEW --output-dir DIR
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, **kwargs)


def read_metadata(dist_info: Path) -> tuple[str, str]:
    name = ""
    version = ""
    for line in (dist_info / "METADATA").read_text().splitlines():
        if line.startswith("Name:") and not name:
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:") and not version:
            version = line.split(":", 1)[1].strip()
        if name and version:
            return name, version
    raise RuntimeError(
        f"Could not parse Name/Version from {dist_info / 'METADATA'}")


def rewrite_metadata(dist_info: Path, old_version: str,
                     new_version: str) -> None:
    metadata = dist_info / "METADATA"
    rewritten = []
    for line in metadata.read_text().splitlines():
        if line.startswith("Version:"):
            line = f"Version: {new_version}"
        elif line.startswith("Requires-Dist:") and f"=={old_version}" in line:
            line = line.replace(f"=={old_version}", f"=={new_version}")
        rewritten.append(line)
    metadata.write_text("\n".join(rewritten) + "\n")


def restamp(wheel: Path, version: str, output_dir: Path) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        run([
            sys.executable, "-m", "wheel", "unpack",
            str(wheel), "-d",
            str(tmpdir)
        ])
        unpacked = next(p for p in tmpdir.iterdir() if p.is_dir())
        dist_info = next(unpacked.glob("*.dist-info"))
        name, old_version = read_metadata(dist_info)
        if old_version != version:
            rewrite_metadata(dist_info, old_version, version)
            new_dist_info = dist_info.with_name(
                dist_info.name.replace(old_version, version, 1))
            if new_dist_info != dist_info:
                dist_info.rename(new_dist_info)
            packed_dir = tmpdir / "packed"
            packed_dir.mkdir()
            run([
                sys.executable, "-m", "wheel", "pack",
                str(unpacked), "-d",
                str(packed_dir)
            ])
            packed = next(packed_dir.glob("*.whl"))
            dest = output_dir / packed.name
            shutil.move(str(packed), dest)
        else:
            dest = output_dir / wheel.name
            shutil.copy2(wheel, dest)
        return dest, name


def verify(wheel: Path, distribution: str, version: str) -> None:
    escaped = version.replace("+", "_")
    if version not in wheel.name and escaped not in wheel.name:
        raise RuntimeError(
            f"Wheel filename {wheel.name} does not contain version {version}")
    with tempfile.TemporaryDirectory() as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.create(venv_dir, with_pip=True)
        python = venv_dir / "bin" / "python"
        run([str(python), "-m", "pip", "install", "--no-deps",
             str(wheel)],
            stdout=subprocess.DEVNULL)
        installed = run([
            str(python), "-c",
            "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))",
            distribution
        ],
                        stdout=subprocess.PIPE).stdout.strip()
        if installed != version:
            raise RuntimeError(
                f"importlib.metadata.version({distribution!r}) is {installed!r}, "
                f"expected {version!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")

    dest, name = restamp(wheel, args.version, args.output_dir.resolve())
    print(f"Re-stamped {wheel.name} -> {dest}")
    if not args.no_verify:
        verify(dest, name, args.version)
        print(f"Verified {name}=={args.version}")
    print(f"wheel_path={dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
