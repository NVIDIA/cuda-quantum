#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Reads package==version lines from stdin and writes a tar archive to stdout
# with the METADATA and the license, copyright, notice and EULA files that the
# installed distributions of those packages ship, as <name>/<file>. Run it with
# the Python whose packages are listed; it needs nothing but the standard library.

import importlib.metadata as md
import io
import re
import sys
import tarfile

PATTERN = re.compile(r"(licen[cs]e|copying|copyright|notice|eula)", re.I)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main():
    wanted = {
        normalize(line.split("==")[0]) for line in sys.stdin.read().split()
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for dist in md.distributions():
            name = dist.metadata["Name"]
            if not name or normalize(name) not in wanted:
                continue
            for f in dist.files or []:
                if not PATTERN.search(f.name) and f.name != "METADATA":
                    continue
                try:
                    data = f.read_binary()
                except Exception:
                    continue
                info = tarfile.TarInfo(
                    normalize(name) + "/" + re.sub(r"/+", "_", str(f)))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
