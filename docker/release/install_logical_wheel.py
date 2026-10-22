# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Install matching Logical and EDEM wheels beside the image's native CUDA-Q."""

import argparse
import base64
import csv
from email.parser import Parser
import hashlib
from importlib.metadata import Distribution
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def pip_install(*args):
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", *args
    ],
                   check=True)


def install(wheel_dir: Path, prefix: Path):
    wheel_dir = wheel_dir.resolve()
    wheels = sorted(wheel_dir.glob("cudaq_logical-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one Logical wheel in {wheel_dir}, "
                         f"found {len(wheels)}")
    if not (prefix / "cudaq/__init__.py").is_file():
        raise ValueError(f"Missing CUDA-Q Python installation in {prefix}")
    if list(prefix.glob("cudaq_logical-*.dist-info")):
        raise ValueError(f"Logical package metadata already exists in {prefix}")
    if ((prefix / "cudaq_edem").exists() or
            list(prefix.glob("cudaq_edem-*.dist-info"))):
        raise ValueError(f"EDEM package already exists in {prefix}")

    with tempfile.TemporaryDirectory(prefix="cudaq-logical-") as temporary:
        staging = Path(temporary) / "package"
        # pip checks wheel compatibility before copying into the existing package.
        pip_install("--no-index", "--no-deps", "--no-compile", "--target",
                    str(staging), str(wheels[0].resolve()))
        metadata_dir, = staging.glob("cudaq_logical-*.dist-info")
        distribution = Distribution.at(metadata_dir)
        # Resolve the native dependency strictly from this run's wheelhouse.
        # Isolated mode ignores user configuration and environment find-links.
        # A fresh target and --ignore-installed prevent a same-version package
        # in the build image from satisfying this request without being copied.
        edem_requirements = [
            requirement for requirement in distribution.requires or []
            if re.match(r"\s*cudaq[-_.]+edem(?![-_.a-z0-9])", requirement,
                        re.IGNORECASE)
        ]
        if len(edem_requirements) != 1:
            raise ValueError("Logical must declare one cudaq-edem dependency")
        pip_install("--isolated", "--no-index", "--no-deps",
                    "--ignore-installed", "--no-compile", "--target",
                    str(staging), "--find-links", str(wheel_dir),
                    edem_requirements[0])
        edem_metadata_dir, = staging.glob("cudaq_edem-*.dist-info")
        directories = ("cudaq/logical", "cudaq/mlir/dialects",
                       "cudaq/mlir/_mlir_libs", "bin", metadata_dir.name,
                       "cudaq_edem", edem_metadata_dir.name)
        for relative in directories:
            if not (staging / relative).is_dir():
                raise ValueError(f"`cudaq.logical` wheel is missing {relative}")

        # Do not ask pip to install core: the native CUDA-Q installation already
        # supplies its libraries and bindings.
        requirements_to_keep = [
            requirement for requirement in distribution.requires or []
            # Match bare or versioned cudaq-core, including equivalent name
            # spellings, but keep different packages such as cudaq-core-tools.
            if not re.match(r"\s*cudaq[-_.]+core(?![-_.a-z0-9])", requirement,
                            re.IGNORECASE)
        ]
        # EDEM is staged beside Logical, where its native RPATH expects it.
        # Keep its metadata dependency, but never fetch another copy from PyPI.
        requirements_to_install = [
            requirement for requirement in requirements_to_keep
            if requirement not in edem_requirements
        ]
        metadata_file = metadata_dir / "METADATA"
        # METADATA uses headers, with one Requires-Dist entry per dependency.
        metadata = Parser().parsestr(metadata_file.read_text(encoding="utf-8"))
        # Remove all dependency entries, then add back only those kept above.
        # Otherwise pip check would still report cudaq-core as missing, even
        # though the native installation supplies it without a core wheel.
        del metadata["Requires-Dist"]
        for requirement in requirements_to_keep:
            metadata["Requires-Dist"] = requirement
        # Change only the extracted copy; the original wheel stays untouched.
        metadata_file.write_text(str(metadata), encoding="utf-8")

        # Install external dependencies without frontend extras.
        if requirements_to_install:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("\n".join(requirements_to_install) + "\n",
                                    encoding="utf-8")
            pip_install("--break-system-packages", "-r", str(requirements))

        # Copy package files, tools, and version metadata without replacing
        # existing CUDA-Q files or claiming them in Logical's installed-file list.
        records = []
        for directory in directories:
            for source in sorted((staging / directory).rglob("*")):
                if not source.is_file() or source in (
                        metadata_dir / "RECORD", metadata_dir /
                        "direct_url.json", edem_metadata_dir / "RECORD",
                        edem_metadata_dir / "direct_url.json"):
                    continue
                relative = source.relative_to(staging)
                destination = prefix / relative
                if destination.exists() or destination.is_symlink():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                with destination.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").digest()
                encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                record = (relative.as_posix(), f"sha256={encoded}",
                          destination.stat().st_size)
                records.append((directory.startswith("cudaq_edem"), record))

        # Record final paths, including bin/qlx-opt and bin/qlx-translate.
        metadata_owners = ((False, metadata_dir), (True, edem_metadata_dir))
        for is_edem, metadata in metadata_owners:
            record = Path(metadata.name) / "RECORD"
            entries = [row for owner, row in records if owner == is_edem]
            entries.append((record.as_posix(), "", ""))
            with (prefix / record).open("w", encoding="utf-8",
                                        newline="") as handle:
                csv.writer(handle).writerows(entries)
        print(f"Installed cudaq-logical {distribution.version} into {prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    arguments = parser.parse_args()
    install(arguments.wheel_dir, arguments.prefix)
