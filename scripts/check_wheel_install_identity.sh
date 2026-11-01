#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Wheel↔install artifact identity: prove the installed SDK files are exactly
# the wheel artifacts' content, per the wheels' own RECORD contract:
#
#   1. every RECORD entry's sha256 digest verifies against the wheel member
#      bytes (wheel-internal integrity);
#   2. every hashed member's installed bytes at <site-packages>/<path> are
#      byte-identical to the wheel member (install↔wheel binding);
#   3. a RECORD entry WITHOUT a digest is only tolerable for the wheel
#      standard's self-record/signature files (.dist-info/RECORD[.jws,.p7s]);
#      any other unhashed member fails closed;
#   4. the installed dist-info METADATA equals the wheel's METADATA (and the
#      dist-info directory exists).
#
# This is the artifact/ABI identity a .comment string cannot give: two
# different implementations cannot share byte identity.
#
# Usage: check_wheel_install_identity.sh <wheel-file> <site-packages>

set -euo pipefail
export LC_ALL=C

if [ $# -ne 2 ] || [ ! -f "$1" ] || [ ! -d "$2" ]; then
    echo "Usage: $0 <wheel-file> <site-packages>" >&2
    exit 2
fi

python3 - "$1" "$2" <<'PYEOF'
import base64, hashlib, os, sys, zipfile

wheel_path, site = sys.argv[1], os.path.realpath(sys.argv[2])

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def record_digest_to_hex(declared):
    padded = declared + "=" * (-len(declared) % 4)
    return base64.urlsafe_b64decode(padded).hex()

fail = 0
def die(msg):
    global fail
    print(f"FAIL: {msg}", file=sys.stderr)
    fail = 1

with zipfile.ZipFile(wheel_path) as z:
    names = set(z.namelist())
    record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
    if record_name is None:
        print(f"FAIL: {os.path.basename(wheel_path)} has no dist-info RECORD")
        sys.exit(1)
    dist_info = record_name.rsplit("/", 1)[0]
    rows = []
    for line in z.read(record_name).decode().splitlines():
        if line:
            rows.append(line.split(","))
    recorded = {parts[0] for parts in rows}
    # Coverage: every ZIP payload member must be covered by a RECORD row.
    # Only explicit directory entries and the wheel standard's
    # self-record/signature files may stand outside it; a member omitted
    # from RECORD silently escapes every digest check.
    for member in sorted(names):
        if member.endswith("/"):
            continue
        if member.endswith((".dist-info/RECORD", ".dist-info/RECORD.jws",
                            ".dist-info/RECORD.p7s")):
            continue
        if member not in recorded:
            print(f"FAIL: wheel member omitted from RECORD: {member}",
                  file=sys.stderr)
            fail = 1
    hashed = [p for p in rows if len(p) > 1 and p[1]]
    if not hashed:
        print("FAIL: RECORD contains no hashed payload members "
              "(vacuous identity evidence)", file=sys.stderr)
        fail = 1
    checked = 0
    for parts in rows:
        path = parts[0]
        declared = parts[1] if len(parts) > 1 else ""
        if path in names:
            member_bytes = z.read(path)
        else:
            print(f"FAIL: RECORD member missing from wheel: {path}", file=sys.stderr)
            fail = 1
            continue
        if not declared:
            # Wheel-standard unhashed rows: the RECORD itself and signatures.
            if path.endswith((".dist-info/RECORD", ".dist-info/RECORD.jws",
                              ".dist-info/RECORD.p7s")):
                continue
            print(f"FAIL: unhashed wheel member outside the signature set: {path}",
                  file=sys.stderr)
            fail = 1
            continue
        if path not in names:
            print(f"FAIL: RECORD row for absent wheel member: {path}", file=sys.stderr)
            fail = 1
            continue
        if not declared.startswith("sha256="):
            print(f"FAIL: unsupported RECORD digest for {path}: {declared}",
                  file=sys.stderr)
            fail = 1
            continue
        wheel_hex = sha256(member_bytes)
        if record_digest_to_hex(declared[len("sha256="):]) != wheel_hex:
            print(f"FAIL: RECORD digest mismatch for {path}", file=sys.stderr)
            fail = 1
            continue
        installed = os.path.join(site, path)
        if not os.path.isfile(installed):
            print(f"FAIL: {path} is in the wheel but not installed", file=sys.stderr)
            fail = 1
            continue
        with open(os.path.realpath(installed), "rb") as fh:
            if sha256(fh.read()) != wheel_hex:
                print(f"FAIL: installed bytes != wheel bytes for {path}", file=sys.stderr)
                fail = 1
                continue
        checked += 1
    # Installed dist-info METADATA must equal the wheel's.
    meta_name = f"{dist_info}/METADATA"
    installed_meta = os.path.join(site, dist_info, "METADATA")
    if not os.path.isfile(installed_meta):
        print(f"FAIL: installed {dist_info}/METADATA missing", file=sys.stderr)
        fail = 1
    else:
        with open(installed_meta, "rb") as fh:
            if fh.read() != z.read(meta_name):
                print(f"FAIL: installed METADATA != wheel METADATA ({dist_info})",
                      file=sys.stderr)
                fail = 1
    if fail:
        sys.exit(1)
    version = next((l.split(": ", 1)[1] for l in z.read(meta_name).decode().splitlines()
                    if l.startswith("Version: ")), "?")
    print(f"check_wheel_install_identity: {checked} files byte-identical "
          f"install↔wheel (RECORD-verified), {dist_info.split('-')[0]}=={version}")
PYEOF
