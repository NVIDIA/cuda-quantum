# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Build the EDEM source distribution and wheel from an isolated source tree."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("dist/edem"))
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="cudaq-edem-build-") as temporary:
        source = Path(temporary)
        for relative in ("edem", "math/gf2"):
            shutil.copytree(root / relative,
                            source / relative,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", ".pytest_cache", "build",
                                "build-*", "_skbuild*", "*.pyc"))
        for name in ("LICENSE", "NOTICE"):
            shutil.copy2(root / name, source / name)
        (source / "LICENSES").mkdir()
        for name in ("LICENSE.nanobind", "LICENSE.robin-map"):
            shutil.copy2(root / "LICENSES" / name, source / "LICENSES" / name)
        configuration = root / "pyproject.toml.edem"
        shutil.copy2(configuration, source / "pyproject.toml.edem")
        shutil.copy2(configuration, source / "pyproject.toml")
        # The default build creates a source distribution, then builds its wheel
        # from that archive.
        subprocess.run([
            sys.executable, "-m", "build", "--outdir",
            str(outdir),
            str(source)
        ],
                       check=True)


if __name__ == "__main__":
    main()
