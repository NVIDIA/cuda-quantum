#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Prints lock entries (the format of cudaq_prereqs.lock) for the third-party
# software that the release images ship but that neither apt, pip, the tpls
# submodules nor install_prerequisites.sh provide:
#   - the MPI stack that docker/build/devdeps.ompi.Dockerfile builds from source
#     (every tarball it downloads, plus the OpenMPI git tag),
#   - uv, copied into the images by docker/build/devcontainer.Dockerfile,
#   - the dependencies CMake fetches at configure time (FetchDependencies.cmake),
#   - the sources of CMake and QRMI, which install_prerequisites.sh only
#     installs as binaries,
#   - with --doca, the DOCA/OFED source bundle that matches DOCA_VERSION (rdma-core,
#     ucx, xpmem, knem and the other Mellanox packages the images install).
#
# The versions are read from the files that build the images, so a component
# that changes there changes here. Usage:
#   scripts/generate_image_sources_lock.py [--doca] >> cudaq_prereqs.lock

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def dockerfile_vars(text: str) -> dict:
    """ENV/ARG assignments with a literal value, e.g. ENV UCX_VERSION=1.21.0."""
    values = {}
    for m in re.finditer(r"^\s*(?:ENV|ARG)\s+(\w+)=([^\s\\]+)", text, re.M):
        values[m.group(1)] = m.group(2).strip('"')
    return values


def expand(url: str, values: dict) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: values[m.group(1)], url)


def name_from_url(url: str) -> str:
    m = re.match(r"https://github\.com/[^/]+/([^/]+)/", url)
    if m:
        return m.group(1).removesuffix(".git")
    return re.match(r"https://[^/]+/.*/([A-Za-z]+)-[0-9]", url).group(1)


def ompi_entries() -> list:
    text = (ROOT / "docker/build/devdeps.ompi.Dockerfile").read_text()
    values = dockerfile_vars(text)
    entries = []
    # Every source tarball the stack is built from. The DOCA package is a .deb.
    for m in re.finditer(r"https://[^\s\"\\]+\.(?:tar\.(?:gz|bz2|xz)|tgz)",
                         text):
        url = expand(m.group(0), values)
        entries.append((name_from_url(url), url))
    # OpenMPI is cloned and built at a tag.
    for m in re.finditer(
            r"git clone (https://github\.com/[^\s]+?)(?:\.git)?\s+\S+.*?git checkout (\S+)",
            text, re.S):
        repo, tag = m.group(1), m.group(2)
        name = {
            "ompi": "openmpi"
        }.get(repo.rsplit("/", 1)[-1],
              repo.rsplit("/", 1)[-1])
        entries.append((name, f"{repo}/archive/refs/tags/{tag}.tar.gz"))
    return entries


def uv_entries() -> list:
    text = (ROOT / "docker/build/devcontainer.Dockerfile").read_text()
    return [(
        "uv",
        f"https://github.com/astral-sh/uv/archive/refs/tags/{m.group(1)}.tar.gz"
    ) for m in re.finditer(r"--from=ghcr\.io/astral-sh/uv:(\S+)", text)]


def fetchcontent_entries() -> list:
    text = (ROOT / "cudaq/lib/Target/cmake/FetchDependencies.cmake").read_text()
    entries = []
    for m in re.finditer(
            r"FetchContent_Declare\(\s*\w+\s+GIT_REPOSITORY\s+(https://github\.com/\S+?)(?:\.git)?\s+(?:#[^\n]*\n\s*)*GIT_TAG\s+(\S+)",
            text):
        repo, ref = m.group(1), m.group(2)
        entries.append((repo.rsplit("/",
                                    1)[-1], f"{repo}/archive/{ref}.tar.gz"))
    return entries


def prereq_source_entries() -> list:
    text = (ROOT / "scripts/prereqs_common.sh").read_text()
    entries = []
    cmake = re.search(r"^CMAKE_VERSION=(\S+)", text, re.M).group(1)
    entries.append((
        "cmake-src",
        f"https://github.com/Kitware/CMake/releases/download/v{cmake}/cmake-{cmake}.tar.gz"
    ))
    repo = re.search(r"^QRMI_RELEASE_REPO=\$\{QRMI_RELEASE_REPO:-([^}]+)\}",
                     text, re.M).group(1)
    tag = re.search(r"^QRMI_RELEASE_TAG=\$\{QRMI_RELEASE_TAG:-([^}]+)\}", text,
                    re.M).group(1)
    entries.append(
        ("qrmi-src",
         f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"))
    return entries


def doca_entries() -> list:
    """The OFED source bundle of the DOCA release the images install."""
    text = (ROOT / "docker/build/devdeps.ompi.Dockerfile").read_text()
    doca = dockerfile_vars(text)["DOCA_VERSION"]  # e.g. 3.1.0-091000-25.07
    release = doca.split("-")[0]
    ofed = doca.rsplit("-", 1)[1]  # 25.07
    index = f"https://linux.mellanox.com/public/repo/doca/{release}/SOURCES/mlnx_ofed/"
    with urllib.request.urlopen(index, timeout=60) as resp:
        html = resp.read().decode()
    bundles = re.findall(
        rf'href="(MLNX_OFED_SRC-debian-{re.escape(ofed)}-[^"]+\.tgz)"', html)
    if not bundles:
        sys.exit(f"No MLNX_OFED_SRC-debian-{ofed} bundle found at {index}")
    bundle = sorted(bundles,
                    key=lambda b: [int(x) for x in re.findall(r"\d+", b)])[-1]
    return [("doca-ofed-src", index + bundle)]


# How NOTICE names the components built into the images.
NOTICE_NAMES = {
    "openmpi": "Open MPI",
    "ucx": "UCX",
    "ucc": "UCC",
    "openpmix": "PMIx",
    "munge": "MUNGE",
    "gdrcopy": "GDRCopy",
    "slurm": "Slurm",
    "uv": "uv",
}


def check_notice(entries) -> int:
    """Every image component must be in NOTICE at the version that is built."""
    notice = (ROOT / "NOTICE").read_text()
    missing = 0
    for name, url in entries:
        if name not in NOTICE_NAMES:
            continue
        version = re.search(r"(\d+(?:\.\d+)+)", url.rsplit("/", 1)[-1]).group(1)
        if f"{NOTICE_NAMES[name]} {version}" not in notice:
            print(
                f"NOTICE has no entry for {NOTICE_NAMES[name]} {version} ({url})",
                file=sys.stderr)
            missing += 1
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Print lock entries for third-party sources the images ship."
    )
    parser.add_argument(
        "--doca",
        action="store_true",
        help="also resolve the DOCA/OFED source bundle (needs network)")
    parser.add_argument(
        "--check-notice",
        action="store_true",
        help="fail if NOTICE lacks an entry for a component at its built version"
    )
    args = parser.parse_args()

    entries = ompi_entries() + uv_entries() + fetchcontent_entries(
    ) + prereq_source_entries()
    if args.check_notice:
        sys.exit(1 if check_notice(entries) else 0)
    if args.doca:
        entries += doca_entries()
    seen = {}
    for name, url in entries:
        if seen.setdefault(name, url) != url:
            sys.exit(
                f"Two different sources for {name}: {seen[name]} and {url}")
    for name, url in sorted(seen.items()):
        print(f"name={name} type=tar url={url}")


if __name__ == "__main__":
    main()
