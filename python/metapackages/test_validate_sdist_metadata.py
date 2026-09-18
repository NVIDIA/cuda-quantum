# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def test_cudaq_sdist_includes_hatch_build_hook():
    """Regression for #3433: sdist must include hatch_build.py for metadata."""
    metapackages = Path(__file__).resolve().parent
    pyproject = metapackages / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    assert '"hatch_build.py"' in contents
    assert (metapackages / "hatch_build.py").is_file()


def test_cudaq_sdist_metadata_name_is_not_unknown():
    """Build an sdist and verify pip can read project name cudaq (issue #3433)."""
    metapackages = Path(__file__).resolve().parent
    readme = metapackages / "README.md.in"
    version_file = metapackages / "_version.txt"
    readme_existed = readme.exists()
    version_existed = version_file.exists()
    readme_backup = readme.read_text(encoding="utf-8") if readme_existed else None
    version_backup = version_file.read_text(encoding="utf-8") if version_existed else None

    try:
        if not readme_existed:
            readme.write_text("# cudaq\n", encoding="utf-8")
        version_file.write_text("0.0.0\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir) / "dist"
            dist_dir.mkdir()
            env = os.environ.copy()
            env["CUDAQ_META_SDIST_BUILD"] = "1"
            subprocess.run(
                [sys.executable, "-m", "build", str(metapackages), "--sdist",
                 "-o", str(dist_dir)],
                check=True,
                env=env,
            )
            sdist = next(dist_dir.glob("cudaq-*.tar.gz"))

            with tarfile.open(sdist, mode="r:gz") as archive:
                names = archive.getnames()
                assert any(n.endswith("hatch_build.py") for n in names), (
                    "hatch_build.py missing from sdist")

            subprocess.run(
                [sys.executable, str(metapackages / "validate_sdist_metadata.py"),
                 str(sdist)],
                check=True,
            )
    finally:
        if readme_existed and readme_backup is not None:
            readme.write_text(readme_backup, encoding="utf-8")
        elif not readme_existed and readme.exists():
            readme.unlink()
        if version_existed and version_backup is not None:
            version_file.write_text(version_backup, encoding="utf-8")
        elif not version_existed and version_file.exists():
            version_file.unlink()
